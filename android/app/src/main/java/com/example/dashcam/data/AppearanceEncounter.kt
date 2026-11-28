package com.ct3d.jolt.data

import androidx.room.Entity

/**
 * Tracks observed vehicle encounter denominators by Make, Body Style, Color, and Color+Body
 * (e.g. "Honda", "SUV", "White SUV") so Jolt's Trends tab can report normalized incident rates
 * (`flagged / observed encounters`) alongside raw flag counts (Requirements §11).
 */
@Entity(
    tableName = "appearance_encounters",
    primaryKeys = ["categoryType", "categoryKey"]
)
data class AppearanceEncounter(
    val categoryType: String, // "MAKE", "BODY", "COLOR", "COLOR_BODY"
    val categoryKey: String,  // e.g. "Honda", "SUV", "White", "White SUV"
    val encounterCount: Int,
    val lastSeenTimestamp: Long
)
