package com.ct3d.jolt.training

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import java.io.File
import java.io.FileOutputStream

/**
 * Phase 6 — Training Data Collector
 *
 * When Training Mode is enabled in the app, this class is called by TelephotoAnalyzer
 * every time the LP detection model fires with confidence ≥ 0.7.
 *
 * Saves two files per capture to filesDir/training_data/<timestamp>/:
 *   - <timestamp>.jpg  — full ROI frame (pre-crop) at full resolution
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
     * @param frameBitmap  Full ROI frame (bottom-65%-cropped) as a Bitmap.
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
