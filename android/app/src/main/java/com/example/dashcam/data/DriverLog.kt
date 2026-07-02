package com.ct3d.jolt.data

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Room Database entity representing a recorded driver event.
 * Contains GPS coords, timestamp, rating (Good/Bad), and the identified
 * vehicle indicators (ALPR Text or Make, Model, Color).
 *
 * Indexed on (plateNormalized, rating) because findBadDriverByPlate() runs on every OCR read
 * (>=2/sec after A5) and filters on exactly those columns — the index avoids a full table scan.
 */
@Entity(
    tableName = "driver_logs",
    indices = [Index(value = ["plateNormalized", "rating"])]
)
data class DriverLog(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val rating: String, // "GOOD" or "BAD"
    val plateOcr: String?, // Raw resolved plate string, shown in the UI verbatim
    val plateNormalized: String? = null, // OCR-confusion-folded form used for known-bad matching (A16)
    val vehicleMmc: String?, // Make, Model, Color if plate was not found
    val timestamp: Long, // Epoch millis
    val latitude: Double, // Device GPS coordinates
    val longitude: Double,
    val batteryLevel: Int = 100, // System battery level percentage (default 100)
    val plateCropPath: String? = null // Absolute path to JPEG crop saved in filesDir/crops/
)

/**
 * Normalize a plate string for confusion-tolerant matching: uppercase, strip non-alphanumerics,
 * then fold the common OCR confusions to a canonical digit (O->0, I->1, B->8, S->5). Store this
 * alongside the raw OCR and match known-bad lookups on it so e.g. a stored "AB0123" still alerts
 * on a read of "ABO123". The UI keeps showing the raw [DriverLog.plateOcr].
 */
fun normalizePlate(raw: String): String =
    raw.uppercase()
        .filter { it.isLetterOrDigit() }
        .map { c -> when (c) { 'O' -> '0'; 'I' -> '1'; 'B' -> '8'; 'S' -> '5'; else -> c } }
        .joinToString("")
