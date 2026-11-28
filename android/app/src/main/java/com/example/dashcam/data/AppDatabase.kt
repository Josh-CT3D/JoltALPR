package com.ct3d.jolt.data

import android.content.Context
import android.util.Log
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

/**
 * Singleton database wrapper over SQLite (version 7).
 *
 * Migrations are explicit (no destructive fallback) so Jack's flagged-driver records survive
 * every schema upgrade across versions 4 -> 5 -> 6 -> 7.
 */
@Database(
    entities = [DriverLog::class, LocationRecord::class, AppearanceEncounter::class],
    version = 7,
    exportSchema = false
)
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

        /**
         * v6 -> v7 (Phase 3): add structured vehicle appearance columns (`vehicleColor`,
         * `vehicleMake`, `vehicleBodyStyle`), `reporterId` for closed Trusted-Group sharing,
         * and the `appearance_encounters` table for normalized incident-rate analytics.
         */
        val MIGRATION_6_7 = object : Migration(6, 7) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE `driver_logs` ADD COLUMN `vehicleColor` TEXT")
                db.execSQL("ALTER TABLE `driver_logs` ADD COLUMN `vehicleMake` TEXT")
                db.execSQL("ALTER TABLE `driver_logs` ADD COLUMN `vehicleBodyStyle` TEXT")
                db.execSQL("ALTER TABLE `driver_logs` ADD COLUMN `reporterId` TEXT NOT NULL DEFAULT 'LOCAL'")
                db.execSQL(
                    "CREATE TABLE IF NOT EXISTS `appearance_encounters` (" +
                        "`categoryType` TEXT NOT NULL, " +
                        "`categoryKey` TEXT NOT NULL, " +
                        "`encounterCount` INTEGER NOT NULL, " +
                        "`lastSeenTimestamp` INTEGER NOT NULL, " +
                        "PRIMARY KEY(`categoryType`, `categoryKey`))"
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
                        .addMigrations(MIGRATION_4_5, MIGRATION_5_6, MIGRATION_6_7)
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
