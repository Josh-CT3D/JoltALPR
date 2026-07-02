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
         * rows get a NULL plateNormalized (they simply won't match confusion-folded lookups until
         * re-flagged). Index names/columns match Room's generated schema (index_<table>_<cols>) so
         * Room's post-migration validation passes. Data preserved.
         */
        val MIGRATION_4_5 = object : Migration(4, 5) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE `driver_logs` ADD COLUMN `plateNormalized` TEXT")
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
