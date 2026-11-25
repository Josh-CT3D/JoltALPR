package com.ct3d.jolt.speech

import android.content.Context
import android.util.Log
import com.ct3d.jolt.data.BehaviorTag
import com.ct3d.jolt.data.JoltSeverity
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import org.json.JSONArray
import org.json.JSONObject
import org.vosk.Model
import org.vosk.Recognizer
import org.vosk.android.RecognitionListener
import org.vosk.android.SpeechService
import org.vosk.android.StorageService

/**
 * Structured hands-free voice commands recognized by Jolt's offline constrained-grammar spotter.
 */
sealed interface VoiceCommand {
    data class FlagVehicle(
        val tags: List<BehaviorTag>,
        val severityOverride: JoltSeverity?,
        val descriptorNotes: String?
    ) : VoiceCommand

    data class AngelVehicle(val descriptorNotes: String?) : VoiceCommand

    data class FollowUpReason(
        val tags: List<BehaviorTag>,
        val severityOverride: JoltSeverity?
    ) : VoiceCommand

    data object UndoOrWrongCar : VoiceCommand
    data object MuteAlerts : VoiceCommand
    data object UnmuteAlerts : VoiceCommand
}

/**
 * Pure, deterministic parser for constrained-grammar Vosk transcripts.
 *
 * Supports:
 *  - One-shot wake commands: "jolt flag white suv weaving and speed", "hey jolt angel that car"
 *  - Two-turn follow-up window (when [awaitingFollowUpReason] = true): driver can just say
 *    "line cutter" or "tailgating" without repeating the wake word "jolt".
 *  - Corrections & controls: "jolt wrong car", "jolt cancel", "jolt mute alerts".
 */
object VoiceCommandParser {

    private val wakeWords = listOf("jolt", "hey jolt", "okay jolt")

    private val descriptorVocabulary = listOf(
        "white", "black", "silver", "gray", "red", "blue", "green",
        "suv", "sedan", "truck", "van", "coupe",
        "honda", "acura", "infiniti", "toyota", "ford", "subaru", "bmw", "tesla", "nissan", "chevy", "jeep"
    )

    fun buildGrammarJson(): String {
        val phrases = linkedSetOf<String>()
        // Wake & command starters
        phrases.addAll(
            listOf(
                "jolt", "hey jolt", "okay jolt",
                "flag", "mark", "angel",
                "that", "the", "car", "vehicle", "ahead", "in front", "on left", "on right", "for", "and",
                "wrong car", "cancel", "undo", "remove last report",
                "mute alerts", "unmute alerts",
                "yellow", "orange", "red", "maroon"
            )
        )
        phrases.addAll(descriptorVocabulary)
        for (tag in BehaviorTag.entries) {
            phrases.addAll(tag.voicePhrases)
        }
        // Escape hatch for non-command cabin speech / road noise
        phrases.add("[unk]")
        return JSONArray(phrases.toList()).toString()
    }

    fun extractTextFromVoskJson(json: String?): String {
        if (json.isNullOrBlank()) return ""
        return try {
            val obj = JSONObject(json)
            (obj.optString("text", "").ifEmpty { obj.optString("partial", "") }).trim()
        } catch (_: Exception) {
            ""
        }
    }

    fun parse(rawTranscript: String, awaitingFollowUpReason: Boolean): VoiceCommand? {
        val cleaned = rawTranscript
            .lowercase()
            .replace("[unk]", " ")
            .replace(Regex("[^a-z ]"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()
        if (cleaned.isEmpty()) return null

        val tokens = cleaned.split(' ')
        val hasWakeWord = tokens.contains("jolt")

        // Global undo/cancel works both with wake word AND bare inside the 6s follow-up window
        if (hasWakeWord || awaitingFollowUpReason) {
            if (cleaned.contains("wrong car") ||
                tokens.contains("cancel") ||
                tokens.contains("undo") ||
                cleaned.contains("remove last")
            ) {
                return VoiceCommand.UndoOrWrongCar
            }
        }

        if (hasWakeWord) {
            if (cleaned.contains("unmute")) return VoiceCommand.UnmuteAlerts
            if (cleaned.contains("mute")) return VoiceCommand.MuteAlerts
        }

        val matchedTags = extractBehaviorTags(cleaned)
        val severityOverride = extractSeverityOverride(tokens, cleaned)
        val descriptorNotes = extractDescriptors(tokens)

        // Follow-up window: no wake word required if the user supplies behavior tag(s) or severity
        if (!hasWakeWord && awaitingFollowUpReason) {
            if (matchedTags.isNotEmpty() || severityOverride != null) {
                return VoiceCommand.FollowUpReason(
                    tags = matchedTags.take(BehaviorTag.MAX_TAGS_PER_INCIDENT),
                    severityOverride = severityOverride
                )
            }
            return null
        }

        if (!hasWakeWord) return null

        // Wake word was spoken!
        if (tokens.contains("angel")) {
            return VoiceCommand.AngelVehicle(descriptorNotes = descriptorNotes)
        }

        val hasFlagIntent = tokens.contains("flag") ||
            tokens.contains("mark") ||
            matchedTags.isNotEmpty() ||
            severityOverride != null

        if (hasFlagIntent) {
            return VoiceCommand.FlagVehicle(
                tags = matchedTags.take(BehaviorTag.MAX_TAGS_PER_INCIDENT),
                severityOverride = severityOverride,
                descriptorNotes = descriptorNotes
            )
        }

        return null
    }

    private fun extractBehaviorTags(cleaned: String): List<BehaviorTag> {
        val padded = " $cleaned "
        val hits = mutableListOf<Pair<Int, BehaviorTag>>()
        for (tag in BehaviorTag.entries) {
            var earliestIdx = Int.MAX_VALUE
            for (phrase in tag.voicePhrases) {
                val idx = padded.indexOf(" $phrase ")
                if (idx in 0 until earliestIdx) {
                    earliestIdx = idx
                }
            }
            if (earliestIdx != Int.MAX_VALUE) {
                hits.add(earliestIdx to tag)
            }
        }
        return hits.sortedBy { it.first }.map { it.second }.distinct()
    }

    private fun extractSeverityOverride(tokens: List<String>, cleaned: String): JoltSeverity? {
        // Avoid treating "red car/suv/truck/sedan" as a severity override unless "mark red" is said
        if (tokens.contains("maroon")) return JoltSeverity.MAROON
        if (tokens.contains("orange")) return JoltSeverity.ORANGE
        if (tokens.contains("yellow")) return JoltSeverity.YELLOW
        if (cleaned.contains("mark red")) return JoltSeverity.RED
        return null
    }

    private fun extractDescriptors(tokens: List<String>): String? {
        val found = tokens.filter { it in descriptorVocabulary }.distinct()
        return if (found.isNotEmpty()) found.joinToString(" ") else null
    }
}

/**
 * Offline always-ready wake-phrase & command controller powered by Vosk (`model-en-us` in assets).
 *
 * Runs at 16 kHz mono PCM on an efficiency core (~3–5% CPU, zero GPU contention with YOLO/MMC),
 * never plays OS recognition chimes, and pauses microphone ingestion whenever TTS speaks.
 */
class VoskVoiceController(
    context: Context,
    private val onCommand: (VoiceCommand, String) -> Unit
) : RecognitionListener {

    enum class State {
        INITIALIZING,
        LISTENING,
        AWAITING_REASON,
        MUTED_BY_USER,
        UNAVAILABLE
    }

    private val appContext = context.applicationContext
    private var model: Model? = null
    private var speechService: SpeechService? = null

    private val _state = MutableStateFlow(State.INITIALIZING)
    val state: StateFlow<State> = _state

    @Volatile
    private var userMuted = false

    @Volatile
    private var tabActive = true

    @Volatile
    private var ttsSpeaking = false

    @Volatile
    private var awaitingReasonUntilMs: Long = 0L

    private var lastDispatchedText = ""
    private var lastDispatchedAtMs = 0L

    fun initialize() {
        if (model != null) {
            syncPauseState()
            return
        }
        _state.value = State.INITIALIZING
        StorageService.unpack(
            appContext,
            MODEL_ASSET_DIR,
            UNPACK_TARGET_DIR,
            { unpackedModel ->
                model = unpackedModel
                startRecognizerService(unpackedModel)
            },
            { exception ->
                Log.w(TAG, "Vosk model unavailable (${exception.message}) — voice commands disabled.")
                _state.value = State.UNAVAILABLE
            }
        )
    }

    private fun startRecognizerService(loadedModel: Model) {
        try {
            val grammar = VoiceCommandParser.buildGrammarJson()
            val recognizer = Recognizer(loadedModel, SAMPLE_RATE, grammar)
            speechService = SpeechService(recognizer, SAMPLE_RATE).also { service ->
                service.startListening(this)
            }
            syncPauseState()
            Log.i(TAG, "Vosk constrained-grammar wake-phrase listener active.")
        } catch (e: Exception) {
            Log.e(TAG, "Failed starting Vosk SpeechService: ${e.message}", e)
            _state.value = State.UNAVAILABLE
        }
    }

    /** Opens a 6-second follow-up window after "Jolt, flag that car" so user can state the reason. */
    fun openFollowUpReasonWindow(durationMs: Long = 6_500L) {
        awaitingReasonUntilMs = System.currentTimeMillis() + durationMs
        updatePublishedState()
    }

    fun closeFollowUpReasonWindow() {
        awaitingReasonUntilMs = 0L
        updatePublishedState()
    }

    fun isAwaitingFollowUpReason(): Boolean = System.currentTimeMillis() < awaitingReasonUntilMs

    fun toggleUserMute(): Boolean {
        userMuted = !userMuted
        syncPauseState()
        return userMuted
    }

    fun setTabActive(active: Boolean) {
        tabActive = active
        syncPauseState()
    }

    fun setTtsSpeaking(speaking: Boolean) {
        ttsSpeaking = speaking
        syncPauseState()
    }

    private fun syncPauseState() {
        val svc = speechService ?: return
        val shouldPause = userMuted || !tabActive || ttsSpeaking
        try {
            svc.setPause(shouldPause)
        } catch (e: Exception) {
            Log.w(TAG, "SpeechService.setPause($shouldPause) failed: ${e.message}")
        }
        updatePublishedState()
    }

    private fun updatePublishedState() {
        if (model == null || speechService == null) return
        _state.value = when {
            userMuted -> State.MUTED_BY_USER
            isAwaitingFollowUpReason() -> State.AWAITING_REASON
            else -> State.LISTENING
        }
    }

    override fun onPartialResult(hypothesis: String?) {
        // Refresh state if follow-up window expired naturally
        if (_state.value == State.AWAITING_REASON && !isAwaitingFollowUpReason()) {
            updatePublishedState()
        }
    }

    override fun onResult(hypothesis: String?) {
        handleFinalHypothesis(hypothesis)
    }

    override fun onFinalResult(hypothesis: String?) {
        handleFinalHypothesis(hypothesis)
    }

    private fun handleFinalHypothesis(hypothesis: String?) {
        if (ttsSpeaking || userMuted || !tabActive) return
        val text = VoiceCommandParser.extractTextFromVoskJson(hypothesis)
        if (text.isEmpty() || text == "[unk]") return

        val now = System.currentTimeMillis()
        if (text == lastDispatchedText && now - lastDispatchedAtMs < 1_200L) return

        val awaiting = isAwaitingFollowUpReason()
        val cmd = VoiceCommandParser.parse(text, awaitingFollowUpReason = awaiting)
        if (cmd != null) {
            lastDispatchedText = text
            lastDispatchedAtMs = now
            Log.i(TAG, "Voice command matched: $cmd (raw='$text', followUp=$awaiting)")
            onCommand(cmd, text)
        } else {
            Log.d(TAG, "Ignored non-wake speech: '$text'")
        }
        updatePublishedState()
    }

    override fun onError(exception: Exception?) {
        Log.e(TAG, "Vosk error: ${exception?.message}", exception)
        _state.value = State.UNAVAILABLE
    }

    override fun onTimeout() {
        closeFollowUpReasonWindow()
    }

    fun release() {
        try {
            speechService?.stop()
            speechService?.shutdown()
            speechService = null
        } catch (_: Exception) {}
        try {
            model?.close()
            model = null
        } catch (_: Exception) {}
    }

    companion object {
        private const val TAG = "VoskVoiceController"
        private const val MODEL_ASSET_DIR = "model-en-us"
        private const val UNPACK_TARGET_DIR = "vosk-model-en-us"
        private const val SAMPLE_RATE = 16000.0f
    }
}
