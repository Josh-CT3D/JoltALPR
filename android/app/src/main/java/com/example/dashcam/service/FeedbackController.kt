package com.ct3d.jolt.service

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Handler
import android.os.Looper
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
 * Eyes-on-road haptic, tone, and offline spoken TextToSpeech alerts (Phase 2 + Phase 2.1).
 *
 * Phase 2.1 Audio Focus Integration:
 *  - Requests transient exclusive/navigation audio focus ([AudioManager.AUDIOFOCUS_GAIN_TRANSIENT])
 *    before speaking OR when the user triggers the "Jolt" wake-word command window.
 *  - This sends `AUDIOFOCUS_LOSS_TRANSIENT` to Spotify / YouTube Music / podcasts so background
 *    music pauses cleanly while Jolt listens or speaks, and automatically resumes playback as soon
 *    as Jolt finishes speaking (or the wake window closes).
 *  - Also notifies [onSpeechStateChanged] whenever TTS begins/ends speaking so [VoskVoiceController]
 *    pauses microphone ingestion during Jolt's own speech — preventing acoustic self-triggers.
 */
class FeedbackController(context: Context) {

    private val appContext = context.applicationContext
    private val mainHandler = Handler(Looper.getMainLooper())
    private val audioManager: AudioManager? =
        appContext.getSystemService(Context.AUDIO_SERVICE) as? AudioManager

    var onSpeechStateChanged: ((Boolean) -> Unit)? = null

    @Volatile
    private var ttsReady = false

    @Volatile
    private var mutedUntilEpochMs: Long = 0L

    @Volatile
    private var lastSpokenAlertText: String? = null

    // Tracks whether audio focus is held for the active "Jolt" wake window and/or active TTS speech
    @Volatile
    private var voiceWindowHoldingFocus = false

    @Volatile
    private var ttsHoldingFocus = false

    private var activeFocusRequest: AudioFocusRequest? = null

    private val speechAudioAttributes: AudioAttributes = AudioAttributes.Builder()
        .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
        .build()

    private val focusReleaseRunnable = Runnable {
        if (!voiceWindowHoldingFocus && !ttsHoldingFocus) {
            abandonAudioFocusInternal()
        }
    }

    // Safety watchdog in case Android's TTS engine ever drops an onDone/onError callback
    private val ttsWatchdogRunnable = Runnable {
        if (ttsHoldingFocus) {
            Log.w(TAG, "TTS utterance watchdog fired — releasing speech state and audio focus.")
            ttsHoldingFocus = false
            onSpeechStateChanged?.invoke(false)
            scheduleAudioFocusReleaseIfIdle()
        }
    }

    private val vibrator: Vibrator? =
        (appContext.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager)
            ?.defaultVibrator

    private val toneGenerator: ToneGenerator? = try {
        // Use STREAM_MUSIC so chimes are audible even when the Pixel ringer switch is on Vibrate.
        ToneGenerator(AudioManager.STREAM_MUSIC, 85)
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
                    engine.setAudioAttributes(speechAudioAttributes)
                    engine.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                        override fun onStart(utteranceId: String?) {
                            ttsHoldingFocus = true
                            onSpeechStateChanged?.invoke(true)
                        }

                        override fun onDone(utteranceId: String?) {
                            onTtsFinished()
                        }

                        @Deprecated("Deprecated in Java")
                        override fun onError(utteranceId: String?) {
                            onTtsFinished()
                        }

                        override fun onError(utteranceId: String?, errorCode: Int) {
                            onTtsFinished()
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

    private fun onTtsFinished() {
        mainHandler.removeCallbacks(ttsWatchdogRunnable)
        ttsHoldingFocus = false
        onSpeechStateChanged?.invoke(false)
        scheduleAudioFocusReleaseIfIdle(delayMs = 220L)
    }

    /**
     * Pauses Spotify / background media while the user is speaking a voice command after "Jolt",
     * so music neither competes with the microphone nor feeds song lyrics into Vosk.
     */
    @Synchronized
    fun setVoiceCommandListeningFocus(active: Boolean) {
        voiceWindowHoldingFocus = active
        if (active) {
            mainHandler.removeCallbacks(focusReleaseRunnable)
            requestTransientAudioFocusInternal(exclusiveForMic = true)
        } else {
            scheduleAudioFocusReleaseIfIdle(delayMs = 250L)
        }
    }

    @Synchronized
    private fun requestTransientAudioFocusInternal(exclusiveForMic: Boolean = false) {
        val am = audioManager ?: return
        if (activeFocusRequest != null) return
        try {
            val gainType = if (exclusiveForMic) {
                AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_EXCLUSIVE
            } else {
                AudioManager.AUDIOFOCUS_GAIN_TRANSIENT
            }
            val req = AudioFocusRequest.Builder(gainType)
                .setAudioAttributes(speechAudioAttributes)
                .setWillPauseWhenDucked(true)
                .setAcceptsDelayedFocusGain(false)
                .setOnAudioFocusChangeListener { /* Transient holder — no action required */ }
                .build()
            val res = am.requestAudioFocus(req)
            if (res == AudioManager.AUDIOFOCUS_REQUEST_GRANTED) {
                activeFocusRequest = req
                Log.d(TAG, "Transient audio focus acquired (exclusive=$exclusiveForMic) — media paused.")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Audio focus request failed: ${e.message}")
        }
    }

    private fun scheduleAudioFocusReleaseIfIdle(delayMs: Long = 180L) {
        mainHandler.removeCallbacks(focusReleaseRunnable)
        mainHandler.postDelayed(focusReleaseRunnable, delayMs)
    }

    @Synchronized
    private fun abandonAudioFocusInternal() {
        val am = audioManager ?: return
        val req = activeFocusRequest ?: return
        activeFocusRequest = null
        try {
            am.abandonAudioFocusRequest(req)
            Log.d(TAG, "Transient audio focus abandoned — media resumed.")
        } catch (e: Exception) {
            Log.w(TAG, "Audio focus abandon failed: ${e.message}")
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

    fun repeatLastAlert() {
        val prev = lastSpokenAlertText
        if (prev != null) {
            speak(prev, forceEvenIfMuted = true)
        } else {
            speak("No recent alert.", forceEvenIfMuted = true)
        }
    }

    /** Short confirmation buzz + ack tone when the user presses FLAG or ANGEL. */
    fun flagPressed() {
        vibrateOneShot(60L)
        playTone(ToneGenerator.TONE_PROP_ACK, 130)
    }

    /** Crisp wake acknowledgment tick + high chime when user says "Jolt" and pauses for command. */
    fun wakeWordAcknowledged() {
        setVoiceCommandListeningFocus(true)
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
        val alertSpeech = reputation.buildSpokenAlert()
        lastSpokenAlertText = alertSpeech
        if (isAlertsMuted()) return
        when (reputation.effectiveSeverity) {
            JoltSeverity.MAROON -> {
                vibrateWaveform(longArrayOf(0, 260, 90, 260, 90, 260))
                playTone(ToneGenerator.TONE_CDMA_EMERGENCY_RINGBACK, 360)
            }
            JoltSeverity.RED -> {
                vibrateWaveform(longArrayOf(0, 200, 110, 200))
                playTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 280)
            }
            JoltSeverity.ORANGE -> {
                vibrateWaveform(longArrayOf(0, 150, 100, 120))
                playTone(ToneGenerator.TONE_PROP_BEEP2, 200)
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
            speak(alertSpeech)
        }
    }

    /** Speaks short interactive voice feedback, pausing Spotify/media for the utterance duration. */
    fun speak(text: String, forceEvenIfMuted: Boolean = false) {
        if (!forceEvenIfMuted && isAlertsMuted()) return
        val engine = tts ?: return
        if (!ttsReady || text.isBlank()) return
        try {
            val id = "jolt_${UUID.randomUUID()}"
            mainHandler.removeCallbacks(focusReleaseRunnable)
            mainHandler.removeCallbacks(ttsWatchdogRunnable)
            ttsHoldingFocus = true
            requestTransientAudioFocusInternal(exclusiveForMic = false)
            onSpeechStateChanged?.invoke(true)
            val res = engine.speak(text, TextToSpeech.QUEUE_FLUSH, null, id)
            if (res == TextToSpeech.ERROR) {
                ttsHoldingFocus = false
                onSpeechStateChanged?.invoke(false)
                scheduleAudioFocusReleaseIfIdle()
            } else {
                mainHandler.postDelayed(ttsWatchdogRunnable, 8_000L)
            }
        } catch (e: Exception) {
            ttsHoldingFocus = false
            onSpeechStateChanged?.invoke(false)
            scheduleAudioFocusReleaseIfIdle()
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
        mainHandler.removeCallbacksAndMessages(null)
        voiceWindowHoldingFocus = false
        ttsHoldingFocus = false
        abandonAudioFocusInternal()
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
