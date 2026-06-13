package com.ct3d.jolt.data

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Room Database entity representing a recorded driver event.
 * Contains GPS coords, timestamp, rating (Good/Bad), and the identified
 * vehicle indicators (ALPR Text or Make, Model, Color).
 */
@Entity(tableName = "driver_logs")
data class DriverLog(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val rating: String, // "GOOD" or "BAD"
    val plateOcr: String?, // Resolved license plate digits
    val vehicleMmc: String?, // Make, Model, Color if plate was not found
    val timestamp: Long, // Epoch millis
    val latitude: Double, // Device GPS coordinates
    val longitude: Double,
    val batteryLevel: Int = 100, // System battery level percentage (default 100)
    val plateCropPath: String? = null // Absolute path to JPEG crop saved in filesDir/crops/
)
