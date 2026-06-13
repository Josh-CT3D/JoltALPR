package com.ct3d.jolt.data

import android.util.Log
import androidx.room.*
import kotlinx.coroutines.flow.Flow

/**
 * Data Access Object for Room database.
 * Facilitates asynchronous data insertions, deletions, and active standard query flows.
 */
@Dao
interface DriverLogDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertLogInternal(log: DriverLog): Long

    @Query("SELECT * FROM driver_logs ORDER BY timestamp DESC")
    fun getAllLogs(): Flow<List<DriverLog>>

    @Delete
    suspend fun deleteLogInternal(log: DriverLog)

    @Query("DELETE FROM driver_logs")
    suspend fun clearAllLogsInternal()

    /**
     * Known Bad Driver Alert lookup.
     * Returns the most recent BAD-rated record matching this exact plate string,
     * or null if the plate has never been flagged.
     * Called every time the OCR pipeline produces a result (1 fps).
     */
    @Query("SELECT * FROM driver_logs WHERE plateOcr = :plateText AND rating = 'BAD' ORDER BY timestamp DESC LIMIT 1")
    suspend fun findBadDriverByPlate(plateText: String): DriverLog?

    // --- Continuous GPS Breadcrumbs Location History ---
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertLocationRecordInternal(record: LocationRecord): Long

    @Query("SELECT * FROM location_history ORDER BY timestamp DESC")
    fun getAllLocationRecords(): Flow<List<LocationRecord>>

    @Query("DELETE FROM location_history")
    suspend fun clearAllLocationRecordsInternal()

    // Deletes breadcrumbs older than the given timestamp cutoff.
    // Called periodically to keep the table from growing unbounded (17k rows/day at 5s interval).
    @Query("DELETE FROM location_history WHERE timestamp < :cutoffMs")
    suspend fun deleteLocationRecordsOlderThan(cutoffMs: Long)
}

// Extension functions or helper wrapper to enable unified, logging-aware database transactions
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
        Log.w("AppDatabase", "WARNING: All DriverLogs successfully cleared from SQLite")
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

