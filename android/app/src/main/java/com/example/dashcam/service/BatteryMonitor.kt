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

    data class Snapshot(
        val levelPct: Int,
        val temperatureC: Float?,
        val isCharging: Boolean
    )

    // Last successfully-resolved level (0..100). Used as a fallback when a single read momentarily
    // fails, so a transient glitch doesn't get persisted as the sentinel value.
    @Volatile
    private var lastKnownGood: Int = BATTERY_UNKNOWN

    /**
     * Returns the current battery level as a percentage (0..100). If the current read fails, falls
     * back to the last-known-good reading; only returns [BATTERY_UNKNOWN] if no read has ever
     * succeeded. Callers should render [BATTERY_UNKNOWN] via [format], not as a raw "-1%".
     */
    fun getBatteryLevel(): Int = getSnapshot().levelPct

    /** Reads battery %, temperature in °C, and charging state in a single broadcast query. */
    fun getSnapshot(): Snapshot {
        return try {
            val batteryStatus: Intent? = context.registerReceiver(
                null,
                IntentFilter(Intent.ACTION_BATTERY_CHANGED)
            )
            val level = batteryStatus?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
            val scale = batteryStatus?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
            val tempTenths = batteryStatus?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
                ?: Int.MIN_VALUE
            val status = batteryStatus?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1

            val pct = if (level >= 0 && scale > 0) {
                val resolved = (level * 100 / scale.toFloat()).toInt().coerceIn(0, 100)
                lastKnownGood = resolved
                resolved
            } else {
                lastKnownGood
            }
            val tempC = if (tempTenths in -400..1200) tempTenths / 10f else null
            val charging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                status == BatteryManager.BATTERY_STATUS_FULL

            Snapshot(levelPct = pct, temperatureC = tempC, isCharging = charging)
        } catch (e: Exception) {
            Log.e("BatteryMonitor", "Failed to read battery snapshot: ${e.localizedMessage}", e)
            Snapshot(levelPct = lastKnownGood, temperatureC = null, isCharging = false)
        }
    }

    companion object {
        /** Sentinel stored when the battery level could never be resolved. */
        const val BATTERY_UNKNOWN = -1

        /** Human-readable battery string: "N/A" for the unknown sentinel, otherwise "<level>%". */
        fun format(level: Int): String = if (level < 0) "N/A" else "$level%"
    }
}
