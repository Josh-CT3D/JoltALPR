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
    /** User spoke the wake word ("Jolt" / "Hey Jolt") and paused — opens an 8s active command window. */
    data object WakeDetected : VoiceCommand

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
 *  - Standalone wake word ("Jolt" / "Hey Jolt") -> [VoiceCommand.WakeDetected] (opens 8s command window)
 *  - One-shot wake commands ("Jolt flag weaving", "Hey Jolt angel that car")
 *  - Active awake window ([isAwakeForCommand] = true): after saying "Jolt", any action word ("flag",
 *    "weaving", "cutting", "angel", "cancel", etc.) executes immediately without repeating "Jolt"!
 *  - Post-flag follow-up window ([awaitingFollowUpReason] = true): refines the just-created flag.
 */
object VoiceCommandParser {

    // Phonetic aliases ("bolt", "volt") catch noisy-cabin consonant shifts on "jolt".
    private val wakeTokens = setOf("jolt", "bolt", "volt")

    private val descriptorVocabulary = listOf(
        "white", "black", "silver", "gray", "red", "blue", "green",
        "suv", "sedan", "truck", "van", "coupe",
        "honda", "acura", "infiniti", "toyota", "ford", "subaru", "bmw", "tesla", "nissan", "chevy", "jeep"
    )

    fun buildGrammarJson(): String {
        val phrases = linkedSetOf<String>()
        phrases.addAll(
            listOf(
                "jolt", "hey jolt", "okay jolt", "hey bolt", "hey volt",
                "flag", "flag that", "flag car", "mark", "angel", "angel car",
                "that", "the", "car", "vehicle", "ahead", "in front", "on left", "on right", "for", "and",
                "wrong car", "cancel", "undo", "delete", "remove", "remove last",
                "mute", "mute alerts", "sound on", "alerts on",
                "yellow", "orange", "red", "maroon"
            )
        )
        phrases.addAll(descriptorVocabulary)
        for (tag in BehaviorTag.entries) {
            phrases.addAll(tag.voicePhrases)
        }
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

    fun containsWakeToken(rawTranscript: String): Boolean {
        val tokens = rawTranscript.lowercase()
            .replace(Regex("[^a-z ]"), " ")
            .split(' ')
            .filter { it.isNotEmpty() }
        return tokens.any { it in wakeTokens }
    }

    fun parse(
        rawTranscript: String,
        awaitingFollowUpReason: Boolean,
        isAwakeForCommand: Boolean = false
    ): VoiceCommand? {
        val cleaned = rawTranscript
            .lowercase()
            .replace("[unk]", " ")
            .replace(Regex("[^a-z ]"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()
        if (cleaned.isEmpty()) return null

        val tokens = cleaned.split(' ')
        val hasWakeWord = tokens.any { it in wakeTokens }
        val activeContext = hasWakeWord || isAwakeForCommand || awaitingFollowUpReason

        // Global undo/cancel works whenever awake, in follow-up, or prefixed with wake word
        if (activeContext) {
            if (cleaned.contains("wrong") ||
                tokens.contains("cancel") ||
                tokens.contains("undo") ||
                tokens.contains("delete") ||
                cleaned.contains("remove")
            ) {
                return VoiceCommand.UndoOrWrongCar
            }
        }

        if (hasWakeWord || isAwakeForCommand) {
            if (cleaned.contains("alerts on") || cleaned.contains("sound on")) {
                return VoiceCommand.UnmuteAlerts
            }
            if (tokens.contains("mute")) {
                return VoiceCommand.MuteAlerts
            }
        }

        val matchedTags = extractBehaviorTags(cleaned)
        val severityOverride = extractSeverityOverride(tokens, cleaned)
        val descriptorNotes = extractDescriptors(tokens)

        // Post-flag follow-up window: attach spoken reason to the flag that was just saved
        if (!hasWakeWord && awaitingFollowUpReason) {
            if (matchedTags.isNotEmpty() || severityOverride != null) {
                return VoiceCommand.FollowUpReason(
                    tags = matchedTags.take(BehaviorTag.MAX_TAGS_PER_INCIDENT),
                    severityOverride = severityOverride
                )
            }
            if (tokens.contains("angel")) {
                return VoiceCommand.AngelVehicle(descriptorNotes = descriptorNotes)
            }
            return null
        }

        if (!hasWakeWord && !isAwakeForCommand) return null

        // Either wake word was spoken in this utterance OR user already said "Jolt" <8s ago!
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

        // User said "Jolt" / "Hey Jolt" by itself (or paused after "Jolt, ...") -> acknowledge & stay awake!
        if (hasWakeWord) {
            return VoiceCommand.WakeDetected
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
 */
class VoskVoiceController(
    context: Context,
    private val onCommand: (VoiceCommand, String) -> Unit
) : RecognitionListener {

    enum class State {
        INITIALIZING,
        LISTENING,
        AWAKE_COMMAND,
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
    private var awakeUntilMs: Long = 0L

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

    /** Opens an 8-second awake command window after hearing "Jolt" so any action word fires immediately. */
    fun openAwakeCommandWindow(durationMs: Long = 8_500L) {
        awakeUntilMs = System.currentTimeMillis() + durationMs
        updatePublishedState()
    }

    fun closeAwakeCommandWindow() {
        awakeUntilMs = 0L
        updatePublishedState()
    }

    fun isAwakeForCommand(): Boolean = System.currentTimeMillis() < awakeUntilMs

    /** Opens a 7-second follow-up window after "Jolt, flag that car" so user can state the reason. */
    fun openFollowUpReasonWindow(durationMs: Long = 7_500L) {
        awaitingReasonUntilMs = System.currentTimeMillis() + durationMs
        awakeUntilMs = 0L
        updatePublishedState()
    }

    fun closeFollowUpReasonWindow() {
        awaitingReasonUntilMs = 0L
        awakeUntilMs = 0L
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
            isAwakeForCommand() -> State.AWAKE_COMMAND
            else -> State.LISTENING
        }
    }

    override fun onPartialResult(hypothesis: String?) {
        if (ttsSpeaking || userMuted || !tabActive) return
        val partial = VoiceCommandParser.extractTextFromVoskJson(hypothesis)
        if (partial.isNotEmpty() && VoiceCommandParser.containsWakeToken(partial)) {
            // Immediately light up the awake state as soon as "Jolt" is heard mid-sentence!
            if (!isAwakeForCommand() && !isAwaitingFollowUpReason()) {
                openAwakeCommandWindow()
            }
        } else if ((_state.value == State.AWAITING_REASON && !isAwaitingFollowUpReason()) ||
            (_state.value == State.AWAKE_COMMAND && !isAwakeForCommand())
        ) {
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
        if (text.isEmpty() || text == "[unk]") {
            updatePublishedState()
            return
        }

        val now = System.currentTimeMillis()
        if (text == lastDispatchedText && now - lastDispatchedAtMs < 1_000L) return

        val awaiting = isAwaitingFollowUpReason()
        val awake = isAwakeForCommand()
        val cmd = VoiceCommandParser.parse(
            rawTranscript = text,
            awaitingFollowUpReason = awaiting,
            isAwakeForCommand = awake
        )
        if (cmd != null) {
            lastDispatchedText = text
            lastDispatchedAtMs = now
            if (cmd is VoiceCommand.WakeDetected) {
                openAwakeCommandWindow()
            } else {
                closeAwakeCommandWindow()
            }
            Log.i(TAG, "Voice command matched: $cmd (raw='$text', awake=$awake, followUp=$awaiting)")
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
