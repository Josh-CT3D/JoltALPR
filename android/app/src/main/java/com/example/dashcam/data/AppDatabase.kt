package com.ct3d.jolt.data

import android.content.Context
import android.util.Log
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

/**
 * Singleton database wrapper over SQLite.
 *
 * Migrations are explicit (no destructive fallback) so Jack's flagged-driver records survive
 * schema bumps. Each version bump adds a Migration(n, n+1); see MIGRATION_4_5.
 */
@Database(entities = [DriverLog::class, LocationRecord::class], version = 5, exportSchema = false)
abstract class AppDatabase : RoomDatabase() {
    abstract fun driverLogDao(): DriverLogDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        /**
         * v4 -> v5: add the plateNormalized column (A16) and create the A8/A16 indices. Existing
         * rows are backfilled (R3): plateNormalized is computed in-SQL to match normalizePlate()
         * exactly — uppercase + the O/0, I/1, B/8, S/5 confusion fold — so drivers Jack flagged
         * before the update keep triggering the known-bad alert (findBadDriverByPlate matches only
         * on plateNormalized). Stored plateOcr tokens are already uppercase alphanumerics (the OCR
         * tokenizer guarantees it), so no non-alphanumeric stripping is needed here. Index
         * names/columns match Room's generated schema (index_<table>_<cols>) so Room's post-migration
         * validation passes. Data preserved.
         */
        val MIGRATION_4_5 = object : Migration(4, 5) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE `driver_logs` ADD COLUMN `plateNormalized` TEXT")
                // R3: backfill the new column for pre-upgrade rows (mirrors normalizePlate()).
                db.execSQL(
                    "UPDATE `driver_logs` SET `plateNormalized` = " +
                        "REPLACE(REPLACE(REPLACE(REPLACE(UPPER(`plateOcr`),'O','0'),'I','1'),'B','8'),'S','5') " +
                        "WHERE `plateOcr` IS NOT NULL"
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS `index_driver_logs_plateNormalized_rating` " +
                        "ON `driver_logs` (`plateNormalized`, `rating`)"
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS `index_location_history_timestamp` " +
                        "ON `location_history` (`timestamp`)"
                )
            }
        }

        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                try {
                    val instance = Room.databaseBuilder(
                        context.applicationContext,
                        AppDatabase::class.java,
                        "driver_behavior_dashcam_db"
                    )
                    .addMigrations(MIGRATION_4_5)
                    .build()
                    INSTANCE = instance
                    Log.i("AppDatabase", "AppDatabase singleton instance created successfully.")
                    instance
                } catch (e: Exception) {
                    Log.e("AppDatabase", "CRITICAL ERROR: Failed to instantiate Room Database builder: ${e.localizedMessage}", e)
                    throw e
                }
            }
        }
    }
}
