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
@Database(entities = [DriverLog::class, LocationRecord::class], version = 6, exportSchema = false)
abstract class AppDatabase : RoomDatabase() {
    abstract fun driverLogDao(): DriverLogDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        /**
         * v4 -> v5: add the plateNormalized column (A16) and create the A8/A16 indices.
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

        /**
         * v5 -> v6 (Phase 2): add behaviorTags, severity, matchConfidence, and notes to driver_logs.
         * Preserves all existing BAD / SIGHTING rows; existing BAD rows default to ORANGE severity
         * and confirmed matchConfidence = 1.0, while legacy GOOD rows map to ANGEL.
         */
        val MIGRATION_5_6 = object : Migration(5, 6) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE `driver_logs` ADD COLUMN `behaviorTags` TEXT")
                db.execSQL("ALTER TABLE `driver_logs` ADD COLUMN `severity` TEXT")
                db.execSQL("ALTER TABLE `driver_logs` ADD COLUMN `matchConfidence` REAL NOT NULL DEFAULT 1.0")
                db.execSQL("ALTER TABLE `driver_logs` ADD COLUMN `notes` TEXT")
                db.execSQL("UPDATE `driver_logs` SET `rating` = 'ANGEL', `severity` = 'ANGEL' WHERE `rating` = 'GOOD'")
                db.execSQL("UPDATE `driver_logs` SET `severity` = 'ORANGE' WHERE `rating` = 'BAD' AND `severity` IS NULL")
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
                    .addMigrations(MIGRATION_4_5, MIGRATION_5_6)
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
