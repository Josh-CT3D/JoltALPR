package com.ct3d.jolt.data

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Room Database entity representing a recorded driver event.
 * Contains GPS coords, timestamp, rating (Good/Bad), and the identified
 * vehicle indicators (ALPR Text or Make, Model, Color).
 *
 * Indexed on (plateOcr, rating) because findBadDriverByPlate() runs on every OCR read
 * (>=2/sec after A5) and filters on exactly those columns — the index avoids a full table scan.
 */
@Entity(
    tableName = "driver_logs",
    indices = [Index(value = ["plateOcr", "rating"])]
)
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
