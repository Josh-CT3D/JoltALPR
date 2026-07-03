package com.ct3d.jolt.training

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.provider.MediaStore
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Phase 6 — Training Data Collector
 *
 * When Training Mode is enabled in the app, this class is called by TelephotoAnalyzer
 * every time the LP detection model fires with confidence ≥ 0.7.
 *
 * Saves two files per capture to filesDir/training_data/<timestamp>/:
 *   - <timestamp>.jpg  — full camera frame (A6 removed the ROI crop) at full resolution
 *   - <timestamp>.txt  — YOLO-format annotation:
 *                          0 <x_center> <y_center> <width> <height>  (all normalized 0-1)
 *                          class 0 = license_plate
 *
 * The resulting folder structure is Roboflow-compatible when zipped by TrainingExportWorker:
 *   images/<timestamp>.jpg
 *   labels/<timestamp>.txt
 *
 * Usage: set [TelephotoAnalyzer.trainingCallback] = collector::onPlateDetected
 */
class TrainingDataCollector(private val context: Context) {

    /**
     * Called by TelephotoAnalyzer when a plate is detected in Training Mode.
     * @param frameBitmap  Full camera frame (A6: no ROI crop) as a Bitmap.
     * @param plateBox     Bounding box of the detected plate within frameBitmap coordinates.
     */
    fun onPlateDetected(frameBitmap: Bitmap, plateBox: RectF) {
        val timestamp = System.currentTimeMillis()
        val dir = File(context.filesDir, "training_data/$timestamp").also { it.mkdirs() }

        try {
            // Save JPEG frame
            val jpegFile = File(dir, "$timestamp.jpg")
            FileOutputStream(jpegFile).use { out ->
                frameBitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
            }

            // Save YOLO annotation (class 0 = license_plate, normalized coords)
            val fw = frameBitmap.width.toFloat()
            val fh = frameBitmap.height.toFloat()
            val xCenter = ((plateBox.left + plateBox.right)  / 2f) / fw
            val yCenter = ((plateBox.top  + plateBox.bottom) / 2f) / fh
            val width   = (plateBox.right  - plateBox.left) / fw
            val height  = (plateBox.bottom - plateBox.top)  / fh

            val labelFile = File(dir, "$timestamp.txt")
            labelFile.writeText("0 $xCenter $yCenter $width $height\n")

            Log.i(TAG, "Training sample saved: $timestamp (box: $plateBox)")

        } catch (e: Exception) {
            Log.e(TAG, "Failed to save training sample: ${e.localizedMessage}", e)
        }
    }

    /** Returns the total number of collected training samples. */
    fun sampleCount(): Int {
        val root = File(context.filesDir, "training_data")
        return root.listFiles()?.count { it.isDirectory } ?: 0
    }

    /**
     * Bundles every collected sample into a Roboflow/YOLO-compatible ZIP
     * (images/<ts>.jpg + labels/<ts>.txt) and writes it to Downloads/ via MediaStore (works on
     * API 31+ with no storage permission). Google Drive's Downloads sync then carries it to the PC.
     * Returns the ZIP file name on success, or null if there was nothing to export or it failed.
     * Runs blocking I/O — call off the main thread.
     */
    fun exportZipToDownloads(): String? {
        val root = File(context.filesDir, "training_data")
        val sampleDirs = root.listFiles()?.filter { it.isDirectory }.orEmpty()
        if (sampleDirs.isEmpty()) {
            Log.w(TAG, "No training samples to export.")
            return null
        }

        val zipName = "jolt_training_${System.currentTimeMillis()}.zip"
        val values = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, zipName)
            put(MediaStore.Downloads.MIME_TYPE, "application/zip")
        }

        val uri = context.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            ?: return null

        return try {
            // R5: a null stream is a real failure — don't fall through and report false success.
            val stream = context.contentResolver.openOutputStream(uri)
                ?: throw IOException("Could not open output stream for $uri")
            stream.use { os ->
                ZipOutputStream(os.buffered()).use { zip ->
                    sampleDirs.forEach { dir ->
                        dir.listFiles()?.forEach { f ->
                            val entryName = when {
                                f.name.endsWith(".jpg") -> "images/${f.name}"
                                f.name.endsWith(".txt") -> "labels/${f.name}"
                                else -> null
                            }
                            if (entryName != null) {
                                zip.putNextEntry(ZipEntry(entryName))
                                f.inputStream().use { it.copyTo(zip) }
                                zip.closeEntry()
                            }
                        }
                    }
                }
            }
            Log.i(TAG, "Exported ${sampleDirs.size} training samples to Downloads/$zipName")
            zipName
        } catch (e: Exception) {
            Log.e(TAG, "Training ZIP export failed: ${e.localizedMessage}", e)
            // R5: delete the MediaStore row we inserted so it doesn't linger as a 0-byte download.
            try { context.contentResolver.delete(uri, null, null) } catch (_: Exception) {}
            null
        }
    }

    /** Deletes all collected training data. */
    fun clearAll() {
        val root = File(context.filesDir, "training_data")
        root.deleteRecursively()
        Log.i(TAG, "Training data cleared.")
    }

    companion object {
        private const val TAG = "TrainingDataCollector"
    }
}
