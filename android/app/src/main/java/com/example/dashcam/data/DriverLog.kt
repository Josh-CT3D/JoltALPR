package com.ct3d.jolt.data

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Room Database entity (v7) representing a recorded vehicle incident, Angel mark, or sighting.
 *
 * Indexed on (plateNormalized, rating) because reputation lookups run on every OCR read.
 */
@Entity(
    tableName = "driver_logs",
    indices = [Index(value = ["plateNormalized", "rating"])]
)
data class DriverLog(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val rating: String, // "BAD", "ANGEL", or "SIGHTING"
    val plateOcr: String?, // Raw resolved plate string, shown in the UI verbatim
    val plateNormalized: String? = null, // OCR-confusion-folded form used for matching (A16)
    val vehicleMmc: String?, // Clean vehicle summary (e.g. "White Honda SUV")
    val timestamp: Long, // Epoch millis
    val latitude: Double, // Device GPS coordinates (0.0 for privacy-stripped peer imports)
    val longitude: Double,
    val batteryLevel: Int = 100,
    val plateCropPath: String? = null, // Absolute path to JPEG crop saved in filesDir/crops/
    val behaviorTags: String? = null, // Up to 2 comma-separated BehaviorTag names
    val severity: String? = null, // "YELLOW", "ORANGE", "RED", "MAROON", or "ANGEL"
    val matchConfidence: Float = 1.0f, // 1.0 = confirmed plate, <1.0 = partial/advisory match
    val notes: String? = null, // Optional spoken/edited vehicle descriptor or feature note
    val vehicleColor: String? = null, // Phase 3 structured color (e.g. "White")
    val vehicleMake: String? = null, // Phase 3 structured make (e.g. "Honda")
    val vehicleBodyStyle: String? = null, // Phase 3 structured body style (e.g. "SUV")
    val reporterId: String = "LOCAL" // "LOCAL" or "Peer:<unit>" for Trusted-Group shared records
)

/**
 * Normalize a plate string for confusion-tolerant matching: uppercase, strip non-alphanumerics,
 * then fold the common OCR confusions to a canonical digit (O->0, I->1, B->8, S->5).
 */
fun normalizePlate(raw: String): String =
    raw.uppercase()
        .filter { it.isLetterOrDigit() }
        .map { c -> when (c) { 'O' -> '0'; 'I' -> '1'; 'B' -> '8'; 'S' -> '5'; else -> c } }
        .joinToString("")
