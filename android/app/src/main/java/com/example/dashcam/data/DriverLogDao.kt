package com.ct3d.jolt.data

import android.util.Log
import androidx.room.*
import kotlinx.coroutines.flow.Flow

/**
 * Data Access Object for Room database (v7).
 */
@Dao
interface DriverLogDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertLogInternal(log: DriverLog): Long

    @Query("SELECT * FROM driver_logs ORDER BY timestamp DESC")
    fun getAllLogs(): Flow<List<DriverLog>>

    @Query("SELECT * FROM driver_logs ORDER BY timestamp DESC")
    suspend fun getAllLogsSnapshot(): List<DriverLog>

    @Delete
    suspend fun deleteLogInternal(log: DriverLog)

    @Query("DELETE FROM driver_logs")
    suspend fun clearAllLogsInternal()

    @Query("SELECT * FROM driver_logs WHERE plateNormalized = :normalizedPlate AND rating = 'BAD' ORDER BY timestamp DESC LIMIT 1")
    suspend fun findBadDriverByPlate(normalizedPlate: String): DriverLog?

    @Query("SELECT * FROM driver_logs WHERE plateNormalized = :normalizedPlate ORDER BY timestamp DESC")
    suspend fun findVehicleHistoryByPlate(normalizedPlate: String): List<DriverLog>

    /** Returns all user/peer incident rows ("BAD" and "ANGEL") for Tier 2/3 partial matching. */
    @Query("SELECT * FROM driver_logs WHERE rating IN ('BAD', 'ANGEL') ORDER BY timestamp DESC")
    suspend fun getAllIncidentCandidates(): List<DriverLog>

    /** Updates behavior tags and severity on an existing incident. */
    @Query("UPDATE driver_logs SET behaviorTags = :behaviorTags, severity = :severity WHERE id = :id")
    suspend fun updateLogTagsAndSeverity(id: Long, behaviorTags: String?, severity: String?)

    /**
     * Phase 2.1 Forward Plate Backfill: attaches a newly confirmed consensus plate + crop JPEG
     * when a highway vehicle was voice-flagged 1–5s before its plate locked OCR consensus.
     */
    @Query(
        "UPDATE driver_logs SET plateOcr = :plateOcr, plateNormalized = :plateNormalized, " +
            "plateCropPath = COALESCE(:plateCropPath, plateCropPath), matchConfidence = :matchConfidence " +
            "WHERE id = :id"
    )
    suspend fun backfillPlateOnRecentFlag(
        id: Long,
        plateOcr: String,
        plateNormalized: String,
        plateCropPath: String?,
        matchConfidence: Float = 1.0f
    )

    /** Backfills structured appearance columns from legacy `vehicleMmc` / `notes` after v6->v7 upgrade. */
    @Query(
        "UPDATE driver_logs SET vehicleColor = :color, vehicleMake = :make, " +
            "vehicleBodyStyle = :bodyStyle, vehicleMmc = :cleanSummary WHERE id = :id"
    )
    suspend fun updateStructuredAppearance(
        id: Long,
        color: String?,
        make: String?,
        bodyStyle: String?,
        cleanSummary: String?
    )

    /** Full parked-review incident edit (Requirements §13 Correctability). */
    @Query(
        "UPDATE driver_logs SET plateOcr = :plateOcr, plateNormalized = :plateNormalized, " +
            "vehicleMmc = :vehicleMmc, vehicleColor = :vehicleColor, vehicleMake = :vehicleMake, " +
            "vehicleBodyStyle = :vehicleBodyStyle, behaviorTags = :behaviorTags, " +
            "severity = :severity, notes = :notes WHERE id = :id"
    )
    suspend fun updateLogFullReview(
        id: Long,
        plateOcr: String?,
        plateNormalized: String?,
        vehicleMmc: String?,
        vehicleColor: String?,
        vehicleMake: String?,
        vehicleBodyStyle: String?,
        behaviorTags: String?,
        severity: String?,
        notes: String?
    )

    @Query("SELECT * FROM driver_logs WHERE rating IN ('BAD', 'ANGEL') ORDER BY timestamp DESC LIMIT 1")
    suspend fun getLatestUserFlag(): DriverLog?

    /** Deduplication check for Trusted-Group bundle import. */
    @Query(
        "SELECT COUNT(*) FROM driver_logs WHERE reporterId = :reporterId AND timestamp = :timestamp " +
            "AND COALESCE(plateNormalized, '') = :plateNormOrEmpty"
    )
    suspend fun countMatchingImportedLog(reporterId: String, timestamp: Long, plateNormOrEmpty: String): Int

    // --- Phase 3 Appearance Encounter Denominators ---
    @Query("SELECT * FROM appearance_encounters")
    fun getAllAppearanceEncounters(): Flow<List<AppearanceEncounter>>

    @Query(
        "SELECT * FROM appearance_encounters WHERE categoryType = :type AND categoryKey = :key LIMIT 1"
    )
    suspend fun getAppearanceEncounter(type: String, key: String): AppearanceEncounter?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsertAppearanceEncounter(entry: AppearanceEncounter)

    @Query("DELETE FROM appearance_encounters")
    suspend fun clearAllAppearanceEncountersInternal()

    // --- Continuous GPS Breadcrumbs Location History ---
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertLocationRecordInternal(record: LocationRecord): Long

    @Query("SELECT * FROM location_history ORDER BY timestamp DESC")
    fun getAllLocationRecords(): Flow<List<LocationRecord>>

    @Query("DELETE FROM location_history")
    suspend fun clearAllLocationRecordsInternal()

    @Query("DELETE FROM location_history WHERE timestamp < :cutoffMs")
    suspend fun deleteLocationRecordsOlderThan(cutoffMs: Long)
}

suspend fun DriverLogDao.incrementEncounter(type: String, key: String, timestamp: Long) {
    if (key.isBlank()) return
    try {
        val existing = getAppearanceEncounter(type, key)
        val nextCount = (existing?.encounterCount ?: 0) + 1
        upsertAppearanceEncounter(
            AppearanceEncounter(
                categoryType = type,
                categoryKey = key,
                encounterCount = nextCount,
                lastSeenTimestamp = timestamp
            )
        )
    } catch (e: Exception) {
        Log.w("AppDatabase", "Encounter increment failed ($type/$key): ${e.localizedMessage}")
    }
}

suspend fun DriverLogDao.safeInsertLog(log: DriverLog): Long {
    return try {
        val result = insertLogInternal(log)
        Log.i("AppDatabase", "Successfully inserted DriverLog [ID: $result] for rating: ${log.rating}")
        result
    } catch (e: Exception) {
        Log.e("AppDatabase", "CRITICAL: Fallback database write failed for DriverLog: ${e.localizedMessage}", e)
        throw e
    }
}

suspend fun DriverLogDao.safeDeleteLog(log: DriverLog) {
    try {
        deleteLogInternal(log)
        Log.i("AppDatabase", "Successfully deleted DriverLog [ID: ${log.id}]")
    } catch (e: Exception) {
        Log.e("AppDatabase", "CRITICAL: Room record deletion failed for ID ${log.id}: ${e.localizedMessage}", e)
    }
}

suspend fun DriverLogDao.safeClearAllLogs() {
    try {
        clearAllLogsInternal()
        clearAllAppearanceEncountersInternal()
        Log.w("AppDatabase", "WARNING: All DriverLogs and encounter counters cleared from SQLite")
    } catch (e: Exception) {
        Log.e("AppDatabase", "CRITICAL: Database clear transaction failed: ${e.localizedMessage}", e)
    }
}

suspend fun DriverLogDao.safeInsertLocationRecord(record: LocationRecord): Long {
    return try {
        val result = insertLocationRecordInternal(record)
        Log.v("AppDatabase", "GPS_BREADCRUMB: Saved breadcrumb of point lat=${record.latitude}, lon=${record.longitude} at ${record.timestamp}")
        result
    } catch (e: Exception) {
        Log.e("AppDatabase", "CRITICAL ERROR: Failed to insert GPS history entry to Room DB: ${e.localizedMessage}", e)
        -1L
    }
}

suspend fun DriverLogDao.safeClearAllLocationRecords() {
    try {
        clearAllLocationRecordsInternal()
        Log.w("AppDatabase", "WARNING: Location history breadcrumbs cleared successfully")
    } catch (e: Exception) {
        Log.e("AppDatabase", "CRITICAL: Failed to clear location history records: ${e.localizedMessage}", e)
    }
}
