package com.ct3d.jolt.data

import android.content.ContentValues
import android.content.Context
import android.provider.MediaStore
import android.util.Log
import java.io.File
import java.io.IOException
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * A21 #4 — full backup of Jack's flagged-driver records to Downloads/, for Google Drive sync.
 *
 * The explicit migrations (MIGRATION_4_5 etc.) protect this data against schema bumps, but nothing
 * protected it against a lost/wiped/replaced phone. This closes that gap.
 *
 * Produces `jolt_backup_<timestamp>.zip`:
 * ```
 *   driver_behavior_dashcam_db     <- the SQLite file, WAL-checkpointed
 *   crops/<timestamp>.jpg          <- the evidence photos DriverLog.plateCropPath points at
 * ```
 *
 * The crops are included deliberately. A DB-only backup restores rows whose plateCropPath refers
 * to files that no longer exist — every History thumbnail silently blank, and the evidence photo
 * is the entire point of the FLAG button.
 *
 * RESTORE IS INTENTIONALLY NOT IMPLEMENTED IN-APP. Overwriting a live Room database while the app
 * holds it open risks corrupting the very records this is meant to protect, and a half-finished
 * restore is worse than no restore. See `restoreInstructions()` for the documented manual path.
 */
class DatabaseBackup(private val context: Context) {

    /**
     * Bundle the database + crop images into Downloads/. Returns the file name on success, or null
     * if there was nothing to back up or it failed. Runs blocking I/O — call off the main thread.
     */
    fun exportBackupToDownloads(db: AppDatabase): String? {
        val dbFile = context.getDatabasePath(DB_NAME)
        if (!dbFile.exists()) {
            Log.w(TAG, "No database file at ${dbFile.absolutePath} — nothing to back up.")
            return null
        }

        // Room runs in WAL mode: recent commits can still live in the -wal sidecar, so copying the
        // main file alone can silently drop the newest flags. TRUNCATE merges the WAL back into the
        // db and empties it, leaving one self-contained file to archive.
        try {
            db.openHelper.writableDatabase.query("PRAGMA wal_checkpoint(TRUNCATE)").use { c ->
                if (c.moveToFirst()) {
                    // columns: busy, log pages, checkpointed pages. busy != 0 means readers blocked it.
                    if (c.getInt(0) != 0) {
                        Log.w(TAG, "WAL checkpoint reported busy — backup may omit the newest rows.")
                    }
                }
            }
        } catch (e: Exception) {
            // Non-fatal: we still archive what's on disk rather than failing the whole backup.
            Log.w(TAG, "WAL checkpoint failed (${e.localizedMessage}) — continuing with file as-is.")
        }

        val cropsDir = File(context.filesDir, "crops")
        val cropFiles = cropsDir.listFiles()?.filter { it.isFile } ?: emptyList()

        val zipName = "jolt_backup_${System.currentTimeMillis()}.zip"
        val values = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, zipName)
            put(MediaStore.Downloads.MIME_TYPE, "application/zip")
        }

        val uri = context.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            ?: run {
                Log.e(TAG, "MediaStore insert returned null — cannot write backup.")
                return null
            }

        return try {
            // R5 pattern: a null stream is a real failure, not something to skip past silently.
            val stream = context.contentResolver.openOutputStream(uri)
                ?: throw IOException("Could not open output stream for $uri")
            stream.use { os ->
                ZipOutputStream(os.buffered()).use { zip ->
                    zip.putNextEntry(ZipEntry(DB_NAME))
                    dbFile.inputStream().use { it.copyTo(zip) }
                    zip.closeEntry()

                    cropFiles.forEach { crop ->
                        zip.putNextEntry(ZipEntry("crops/${crop.name}"))
                        crop.inputStream().use { it.copyTo(zip) }
                        zip.closeEntry()
                    }
                }
            }
            Log.i(TAG, "Backup written: Downloads/$zipName " +
                       "(db ${dbFile.length() / 1024}KB + ${cropFiles.size} crops)")
            zipName
        } catch (e: Exception) {
            Log.e(TAG, "Backup failed: ${e.localizedMessage}", e)
            // R5 pattern: drop the placeholder row so a 0-byte "backup" can't masquerade as real.
            try { context.contentResolver.delete(uri, null, null) } catch (_: Exception) {}
            null
        }
    }

    /** Count of records that would be included, for the UI label. */
    fun cropCount(): Int = File(context.filesDir, "crops").listFiles()?.count { it.isFile } ?: 0

    companion object {
        private const val TAG = "DatabaseBackup"
        private const val DB_NAME = "driver_behavior_dashcam_db"

        /**
         * Manual restore procedure (deliberately not automated — see class kdoc).
         *
         * With the app FORCE-STOPPED, from a machine with adb:
         * ```
         *   unzip jolt_backup_<ts>.zip -d restore/
         *   adb push restore/driver_behavior_dashcam_db /sdcard/
         *   adb shell run-as com.ct3d.jolt cp /sdcard/driver_behavior_dashcam_db \
         *       /data/data/com.ct3d.jolt/databases/driver_behavior_dashcam_db
         *   # remove stale sidecars so SQLite doesn't replay an old WAL over the restored file
         *   adb shell run-as com.ct3d.jolt rm -f \
         *       /data/data/com.ct3d.jolt/databases/driver_behavior_dashcam_db-wal \
         *       /data/data/com.ct3d.jolt/databases/driver_behavior_dashcam_db-shm
         *   # crops must go back too or every History thumbnail restores blank
         *   adb push restore/crops/. /sdcard/crops/
         *   adb shell run-as com.ct3d.jolt cp -r /sdcard/crops/. \
         *       /data/data/com.ct3d.jolt/files/crops/
         * ```
         * plateCropPath stores absolute paths, so this only round-trips onto the same package on
         * the same user profile. Moving to a different device/profile needs a path rewrite.
         */
        fun restoreInstructions(): String = "See DatabaseBackup.restoreInstructions() kdoc"
    }
}
