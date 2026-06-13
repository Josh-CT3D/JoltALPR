package com.ct3d.jolt.data

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Room database entity storing a high-density GPS track point
 * for continuous location recording/breadcrumbs mapping.
 */
@Entity(tableName = "location_history")
data class LocationRecord(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    val timestamp: Long, // Epoch millis
    val latitude: Double,
    val longitude: Double,
    val speed: Float,     // Speed in m/s (0.0 if not available)
    val accuracy: Float   // Accuracy in meters
)
