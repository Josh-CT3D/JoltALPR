package com.ct3d.jolt.service

import android.content.Context
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.VibrationEffect
import android.os.VibratorManager
import android.os.Vibrator
import android.util.Log

/**
 * Eyes-on-road haptic + sound cues (A21). Vibration uses VibratorManager (min SDK 31) and the
 * VIBRATE permission (normal, auto-granted); tones use ToneGenerator (no permission). A single
 * ToneGenerator is reused and released in [release].
 *
 * Failures are swallowed and logged — feedback is a nicety, never worth crashing a drive over.
 */
class FeedbackController(context: Context) {

    private val appContext = context.applicationContext

    private val vibrator: Vibrator? =
        (appContext.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager)
            ?.defaultVibrator

    private val toneGenerator: ToneGenerator? = try {
        ToneGenerator(AudioManager.STREAM_NOTIFICATION, 90)
    } catch (e: Exception) {
        Log.w(TAG, "ToneGenerator unavailable: ${e.message}")
        null
    }

    /** Short confirmation buzz + ack tone when the user presses FLAG. */
    fun flagPressed() {
        vibrateOneShot(60L)
        playTone(ToneGenerator.TONE_PROP_ACK, 150)
    }

    /** Stronger double-buzz + alert tone when a known-bad driver is newly detected. */
    fun badDriverAlert() {
        vibrateWaveform(longArrayOf(0, 200, 120, 200))
        playTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 400)
    }

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
        try { toneGenerator?.release() } catch (e: Exception) { /* ignore */ }
    }

    companion object {
        private const val TAG = "FeedbackController"
    }
}
