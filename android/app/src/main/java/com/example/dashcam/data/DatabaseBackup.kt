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

        val snapshot = createSnapshot(db, dbFile)

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
                    // Entry name stays DB_NAME regardless of where the snapshot came from, so the
                    // archive layout is identical on both the VACUUM and fallback paths.
                    zip.putNextEntry(ZipEntry(DB_NAME))
                    snapshot.inputStream().use { it.copyTo(zip) }
                    zip.closeEntry()

                    cropFiles.forEach { crop ->
                        zip.putNextEntry(ZipEntry("crops/${crop.name}"))
                        crop.inputStream().use { it.copyTo(zip) }
                        zip.closeEntry()
                    }
                }
            }
            Log.i(TAG, "Backup written: Downloads/$zipName " +
                       "(db ${snapshot.length() / 1024}KB + ${cropFiles.size} crops)")
            zipName
        } catch (e: Exception) {
            Log.e(TAG, "Backup failed: ${e.localizedMessage}", e)
            // R5 pattern: drop the placeholder row so a 0-byte "backup" can't masquerade as real.
            try { context.contentResolver.delete(uri, null, null) } catch (_: Exception) {}
            null
        } finally {
            if (snapshot != dbFile) snapshot.delete()
        }
    }

    /**
     * C4: produce a consistent copy of the database to archive.
     *
     * Prefers SQLite's `VACUUM INTO`, which runs inside a read transaction — so the output is a
     * true point-in-time snapshot even though location breadcrumbs keep inserting every 5s, and it
     * compacts the file as a bonus. (Available since SQLite 3.32; minSdk 31 ships well past that.)
     *
     * Falls back to checkpoint-then-copy-in-place. That's nearly always fine — after
     * `wal_checkpoint(TRUNCATE)` new writes go to a fresh WAL and the main file only changes at the
     * next checkpoint — but it isn't consistent *by construction*, which is why VACUUM is preferred.
     *
     * @return the file to archive: a temp snapshot in cacheDir, or [dbFile] itself on the fallback
     *         path. Callers must delete the result if it differs from [dbFile].
     */
    private fun createSnapshot(db: AppDatabase, dbFile: File): File {
        val tmp = File(context.cacheDir, "jolt_backup_snapshot.db")
        tmp.delete() // VACUUM INTO refuses to write to an existing file
        return try {
            // Path is app-internal (no quotes possible), but escape anyway rather than rely on it.
            val escaped = tmp.absolutePath.replace("'", "''")
            db.openHelper.writableDatabase.execSQL("VACUUM INTO '$escaped'")
            if (!tmp.exists() || tmp.length() == 0L) throw IOException("VACUUM INTO produced no file")
            Log.i(TAG, "Snapshot via VACUUM INTO (${tmp.length() / 1024}KB)")
            tmp
        } catch (e: Exception) {
            Log.w(TAG, "VACUUM INTO unavailable (${e.localizedMessage}) — checkpoint+copy fallback")
            tmp.delete()
            checkpointWal(db)
            dbFile
        }
    }

    /**
     * Fold the -wal sidecar back into the main database. Only needed on the fallback path: without
     * it, recent commits live in the sidecar and copying the .db alone silently drops the newest
     * flags. TRUNCATE also empties the WAL, leaving one self-contained file.
     */
    private fun checkpointWal(db: AppDatabase) {
        try {
            db.openHelper.writableDatabase.query("PRAGMA wal_checkpoint(TRUNCATE)").use { c ->
                // columns: busy, log pages, checkpointed pages. busy != 0 means readers blocked it.
                if (c.moveToFirst() && c.getInt(0) != 0) {
                    Log.w(TAG, "WAL checkpoint reported busy — backup may omit the newest rows.")
                }
            }
        } catch (e: Exception) {
            // Non-fatal: archive what's on disk rather than failing the whole backup.
            Log.w(TAG, "WAL checkpoint failed (${e.localizedMessage}) — continuing with file as-is.")
        }
    }

    companion object {
        private const val TAG = "DatabaseBackup"
        private const val DB_NAME = "driver_behavior_dashcam_db"

        /**
         * Manual restore procedure (deliberately not automated — see class kdoc).
         *
         * C3: staged through `/data/local/tmp`, NOT `/sdcard`. Files adb-pushed to shared storage
         * are owned by shell/media_rw, and this app holds no storage permission — so the `run-as`
         * process (running as the app uid) frequently cannot read them back. `/data/local/tmp` is
         * world-traversable and the pushed files land 0644, which the app uid can read.
         *
         * With the app FORCE-STOPPED, from a machine with adb:
         * ```
         *   unzip jolt_backup_<ts>.zip -d restore/
         *
         *   adb push restore/driver_behavior_dashcam_db /data/local/tmp/
         *   adb shell run-as com.ct3d.jolt cp /data/local/tmp/driver_behavior_dashcam_db \
         *       /data/data/com.ct3d.jolt/databases/driver_behavior_dashcam_db
         *
         *   # Delete stale sidecars, or SQLite replays an old WAL over the restored file.
         *   adb shell run-as com.ct3d.jolt rm -f \
         *       /data/data/com.ct3d.jolt/databases/driver_behavior_dashcam_db-wal \
         *       /data/data/com.ct3d.jolt/databases/driver_behavior_dashcam_db-shm
         *
         *   # Crops must go back too, or every History thumbnail restores blank.
         *   adb push restore/crops /data/local/tmp/
         *   adb shell run-as com.ct3d.jolt mkdir -p /data/data/com.ct3d.jolt/files/crops
         *   adb shell run-as com.ct3d.jolt cp -r /data/local/tmp/crops/. \
         *       /data/data/com.ct3d.jolt/files/crops/
         *
         *   adb shell rm -rf /data/local/tmp/crops /data/local/tmp/driver_behavior_dashcam_db
         * ```
         * `plateCropPath` stores ABSOLUTE paths, so this only round-trips onto the same package on
         * the same user profile. A different device/profile needs a path rewrite pass first.
         *
         * REHEARSE THIS ONCE on a throwaway backup before you ever need it for real — an untested
         * restore is a hope, not a backup.
         */
        fun restoreInstructions(): String = "See DatabaseBackup.restoreInstructions() kdoc"
    }
}
