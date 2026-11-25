package com.ct3d.jolt.service

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import com.ct3d.jolt.data.JoltSeverity
import com.ct3d.jolt.data.VehicleReputation
import java.util.Locale
import java.util.UUID

/**
 * Eyes-on-road haptic, tone, and offline spoken TextToSpeech alerts (Phase 2).
 *
 * Notifies [onSpeechStateChanged] whenever TTS begins/ends speaking so [VoskVoiceController]
 * pauses microphone ingestion during Jolt's own speech — preventing acoustic self-triggers.
 */
class FeedbackController(context: Context) {

    private val appContext = context.applicationContext

    var onSpeechStateChanged: ((Boolean) -> Unit)? = null

    @Volatile
    private var ttsReady = false

    @Volatile
    private var mutedUntilEpochMs: Long = 0L

    private val vibrator: Vibrator? =
        (appContext.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager)
            ?.defaultVibrator

    private val toneGenerator: ToneGenerator? = try {
        // Use STREAM_MUSIC so chimes are audible even when the Pixel ringer switch is on Vibrate.
        ToneGenerator(AudioManager.STREAM_MUSIC, 90)
    } catch (e: Exception) {
        Log.w(TAG, "ToneGenerator unavailable: ${e.message}")
        null
    }

    private var tts: TextToSpeech? = null

    init {
        try {
            tts = TextToSpeech(appContext) { status ->
                if (status == TextToSpeech.SUCCESS) {
                    val engine = tts ?: return@TextToSpeech
                    val langRes = engine.setLanguage(Locale.US)
                    engine.setSpeechRate(1.08f) // Crisp, tactical cadence
                    engine.setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build()
                    )
                    engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                        override fun onStart(utteranceId: String?) {
                            onSpeechStateChanged?.invoke(true)
                        }
                        override fun onDone(utteranceId: String?) {
                            onSpeechStateChanged?.invoke(false)
                        }
                        @Deprecated("Deprecated in Java")
                        override fun onError(utteranceId: String?) {
                            onSpeechStateChanged?.invoke(false)
                        }
                        override fun onError(utteranceId: String?, errorCode: Int) {
                            onSpeechStateChanged?.invoke(false)
                        }
                    })
                    ttsReady = langRes != TextToSpeech.LANG_MISSING_DATA &&
                        langRes != TextToSpeech.LANG_NOT_SUPPORTED
                    Log.i(TAG, "TextToSpeech initialized (ready=$ttsReady)")
                } else {
                    Log.w(TAG, "TextToSpeech init failed with status=$status")
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "TextToSpeech unavailable: ${e.message}")
        }
    }

    fun isAlertsMuted(): Boolean = System.currentTimeMillis() < mutedUntilEpochMs

    fun muteAlertsForTenMinutes() {
        mutedUntilEpochMs = System.currentTimeMillis() + 10 * 60 * 1_000L
        speak("Alerts muted for ten minutes.", forceEvenIfMuted = true)
    }

    fun unmuteAlerts() {
        mutedUntilEpochMs = 0L
        speak("Alerts active.", forceEvenIfMuted = true)
    }

    /** Short confirmation buzz + ack tone when the user presses FLAG or ANGEL. */
    fun flagPressed() {
        vibrateOneShot(60L)
        playTone(ToneGenerator.TONE_PROP_ACK, 130)
    }

    /** Crisp wake acknowledgment tick + high chime when user says "Jolt" and pauses for command. */
    fun wakeWordAcknowledged() {
        vibrateOneShot(40L)
        playTone(ToneGenerator.TONE_PROP_BEEP, 110)
    }

    /** Legacy fallback alert cue. */
    fun badDriverAlert() {
        if (isAlertsMuted()) return
        vibrateWaveform(longArrayOf(0, 200, 120, 200))
        playTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 350)
    }

    /**
     * Phase 2 severity-graded haptic + spoken TTS callout on repeat vehicle encounter.
     * Examples: "Red. Weaver.", "Maroon. Repeat line cutter. 2 prior flags.", "Angel vehicle ahead."
     */
    fun announceEncounter(reputation: VehicleReputation) {
        if (isAlertsMuted()) return
        when (reputation.effectiveSeverity) {
            JoltSeverity.MAROON -> {
                vibrateWaveform(longArrayOf(0, 260, 90, 260, 90, 260))
                playTone(ToneGenerator.TONE_CDMA_EMERGENCY_RINGBACK, 420)
            }
            JoltSeverity.RED -> {
                vibrateWaveform(longArrayOf(0, 200, 110, 200))
                playTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 320)
            }
            JoltSeverity.ORANGE -> {
                vibrateWaveform(longArrayOf(0, 150, 100, 120))
                playTone(ToneGenerator.TONE_PROP_BEEP2, 220)
            }
            JoltSeverity.YELLOW -> {
                // Mild nuisance: subtle single haptic tap, no harsh alarm tone (Requirements §9)
                vibrateOneShot(80L)
            }
            JoltSeverity.ANGEL -> {
                vibrateOneShot(45L)
                playTone(ToneGenerator.TONE_PROP_ACK, 140)
            }
        }
        // Yellow alerts stay low-fatigue (visual + light haptic unless repeat); Orange/Red/Maroon/Angel speak.
        if (reputation.effectiveSeverity != JoltSeverity.YELLOW || reputation.badIncidentCount >= 2) {
            speak(reputation.buildSpokenAlert())
        }
    }

    /** Speaks short interactive voice feedback (e.g. "LT 2 3 7 9. Reason?" or "Saved. Weaver."). */
    fun speak(text: String, forceEvenIfMuted: Boolean = false) {
        if (!forceEvenIfMuted && isAlertsMuted()) return
        val engine = tts ?: return
        if (!ttsReady || text.isBlank()) return
        try {
            val id = "jolt_${UUID.randomUUID()}"
            onSpeechStateChanged?.invoke(true)
            val res = engine.speak(text, TextToSpeech.QUEUE_FLUSH, null, id)
            if (res == TextToSpeech.ERROR) {
                onSpeechStateChanged?.invoke(false)
            }
        } catch (e: Exception) {
            onSpeechStateChanged?.invoke(false)
            Log.w(TAG, "TTS speak failed: ${e.message}")
        }
    }

    /** Formats a license plate so TTS reads individual characters cleanly ("LT2379" -> "L T 2 3 7 9"). */
    fun formatPlateForSpeech(plate: String): String =
        plate.uppercase().filter { it.isLetterOrDigit() }.toCharArray().joinToString(" ")

    private fun vibrateOneShot(ms: Long) {
        try {
            vibrator?.takeIf { it.hasVibrator() }
                ?.vibrate(VibrationEffect.createOneShot(ms, VibrationEffect.DEFAULT_AMPLITUDE))
        } catch (e: Exception) {
            Log.w(TAG, "Vibrate failed: ${e.message}")
        }
    }

    private fun vibrateWaveform(pattern: LongArray) {
        try {
            vibrator?.takeIf { it.hasVibrator() }
                ?.vibrate(VibrationEffect.createWaveform(pattern, -1))
        } catch (e: Exception) {
            Log.w(TAG, "Vibrate failed: ${e.message}")
        }
    }

    private fun playTone(toneType: Int, durationMs: Int) {
        try {
            toneGenerator?.startTone(toneType, durationMs)
        } catch (e: Exception) {
            Log.w(TAG, "Tone failed: ${e.message}")
        }
    }

    fun release() {
        try { toneGenerator?.release() } catch (_: Exception) {}
        try {
            tts?.stop()
            tts?.shutdown()
            tts = null
            ttsReady = false
        } catch (_: Exception) {}
    }

    companion object {
        private const val TAG = "FeedbackController"
    }
}
