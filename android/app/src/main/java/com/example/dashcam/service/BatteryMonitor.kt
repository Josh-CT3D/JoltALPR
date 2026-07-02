package com.ct3d.jolt.service

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.util.Log

/**
 * Android system hardware service helper to query current battery metrics.
 * Retrieves real-time battery level percentage using the system Intent.ACTION_BATTERY_CHANGED broadcast.
 */
class BatteryMonitor(private val context: Context) {

    // Last successfully-resolved level (0..100). Used as a fallback when a single read momentarily
    // fails, so a transient glitch doesn't get persisted as the sentinel value.
    @Volatile
    private var lastKnownGood: Int = BATTERY_UNKNOWN

    /**
     * Returns the current battery level as a percentage (0..100). If the current read fails, falls
     * back to the last-known-good reading; only returns [BATTERY_UNKNOWN] if no read has ever
     * succeeded. Callers should render [BATTERY_UNKNOWN] via [format], not as a raw "-1%".
     */
    fun getBatteryLevel(): Int {
        val read = readBatteryLevel()
        return if (read in 0..100) {
            lastKnownGood = read
            read
        } else {
            Log.w("BatteryMonitor", "Battery read failed; falling back to last-known-good=$lastKnownGood")
            lastKnownGood
        }
    }

    /** Raw system read. Returns [BATTERY_UNKNOWN] on any failure. */
    private fun readBatteryLevel(): Int {
        return try {
            val batteryStatus: Intent? = context.registerReceiver(
                null,
                IntentFilter(Intent.ACTION_BATTERY_CHANGED)
            )
            val level = batteryStatus?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
            val scale = batteryStatus?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1

            if (level >= 0 && scale > 0) {
                val percentage = (level * 100 / scale.toFloat()).toInt().coerceIn(0, 100)
                Log.i("BatteryMonitor", "Resolved system battery level percentage: $percentage%")
                percentage
            } else {
                Log.w("BatteryMonitor", "Invalid battery level metrics received: level=$level, scale=$scale")
                BATTERY_UNKNOWN
            }
        } catch (e: Exception) {
            Log.e("BatteryMonitor", "CRITICAL ERROR: Failed to register receiver or parse battery stats: ${e.localizedMessage}", e)
            BATTERY_UNKNOWN
        }
    }

    companion object {
        /** Sentinel stored when the battery level could never be resolved. */
        const val BATTERY_UNKNOWN = -1

        /** Human-readable battery string: "N/A" for the unknown sentinel, otherwise "<level>%". */
        fun format(level: Int): String = if (level < 0) "N/A" else "$level%"
    }
}
