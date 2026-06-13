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

    /**
     * Queries the system status and returns the current battery level as a percentage (0 to 100).
     * Returns -1 if the battery level is unavailable or failed to resolve.
     */
    fun getBatteryLevel(): Int {
        return try {
            val batteryStatus: Intent? = context.registerReceiver(
                null,
                IntentFilter(Intent.ACTION_BATTERY_CHANGED)
            )
            val level = batteryStatus?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
            val scale = batteryStatus?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
            
            if (level >= 0 && scale > 0) {
                val percentage = (level * 100 / scale.toFloat()).toInt()
                Log.i("BatteryMonitor", "Resolved system battery level percentage: $percentage%")
                percentage
            } else {
                Log.w("BatteryMonitor", "Invalid battery level metrics received: level=$level, scale=$scale")
                -1
            }
        } catch (e: Exception) {
            Log.e("BatteryMonitor", "CRITICAL ERROR: Failed to register receiver or parse battery stats: ${e.localizedMessage}", e)
            -1
        }
    }
}
