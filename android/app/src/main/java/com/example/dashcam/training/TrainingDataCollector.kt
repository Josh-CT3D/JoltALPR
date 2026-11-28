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
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Phase 6 + Phase 2.1 — Asynchronous Training Data Collector
 *
 * Saves full-frame JPEGs + multi-box YOLO annotations on a dedicated background I/O thread so
 * JPEG compression (`~80-110ms`) never stalls `jolt-pipeline` during road drives.
 */
class TrainingDataCollector(private val context: Context) {

    @Volatile
    private var lastSavedAtMs: Long = 0L

    @Volatile
    private var lastSavedBox: RectF? = null

    private val ioExecutor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "jolt-training-writer").apply { isDaemon = true }
    }

    /**
     * Checks deduplication synchronously on the caller thread, copies [frameBitmap] ONLY when a
     * sample is accepted, and dispatches JPEG compression + disk write asynchronously so
     * `jolt-pipeline` returns immediately.
     *
     * @return true if accepted and queued for background save, false if deduplicated/throttled.
     */
    fun onPlateDetected(frameBitmap: Bitmap, plateBoxes: List<RectF>): Boolean {
        if (plateBoxes.isEmpty() || frameBitmap.isRecycled) return false

        val now = System.currentTimeMillis()
        val primaryBox = plateBoxes.first()
        val prevBox = lastSavedBox
        val elapsed = now - lastSavedAtMs
        if (elapsed < MIN_SAMPLE_INTERVAL_MS) return false
        if (elapsed < DEDUP_WINDOW_MS && prevBox != null && calculateIoU(prevBox, primaryBox) > DEDUP_IOU_THRESHOLD) {
            return false
        }

        val frameCopy = try {
            frameBitmap.copy(frameBitmap.config ?: Bitmap.Config.ARGB_8888, false)
        } catch (e: Exception) {
            Log.w(TAG, "Could not copy frame for training sample: ${e.message}")
            null
        } ?: return false

        val boxesCopy = plateBoxes.map { RectF(it) }
        lastSavedAtMs = now
        lastSavedBox = RectF(primaryBox)

        try {
            ioExecutor.execute {
                val dir = File(context.filesDir, "training_data/$now").also { it.mkdirs() }
                try {
                    val jpegFile = File(dir, "$now.jpg")
                    FileOutputStream(jpegFile).use { out ->
                        frameCopy.compress(Bitmap.CompressFormat.JPEG, 92, out)
                    }

                    val fw = frameCopy.width.toFloat()
                    val fh = frameCopy.height.toFloat()
                    val yoloLines = boxesCopy.joinToString("\n") { box ->
                        val xCenter = (((box.left + box.right) / 2f) / fw).coerceIn(0f, 1f)
                        val yCenter = (((box.top + box.bottom) / 2f) / fh).coerceIn(0f, 1f)
                        val width   = ((box.right - box.left) / fw).coerceIn(0f, 1f)
                        val height  = ((box.bottom - box.top) / fh).coerceIn(0f, 1f)
                        "0 $xCenter $yCenter $width $height"
                    } + "\n"

                    val labelFile = File(dir, "$now.txt")
                    labelFile.writeText(yoloLines)
                    Log.i(TAG, "Training sample saved async: $now (${boxesCopy.size} boxes)")
                } catch (e: Exception) {
                    Log.e(TAG, "Failed to save training sample: ${e.localizedMessage}", e)
                    dir.deleteRecursively()
                } finally {
                    if (!frameCopy.isRecycled) frameCopy.recycle()
                }
            }
            return true
        } catch (e: Exception) {
            if (!frameCopy.isRecycled) frameCopy.recycle()
            return false
        }
    }

    private fun calculateIoU(a: RectF, b: RectF): Float {
        val l = maxOf(a.left, b.left)
        val t = maxOf(a.top, b.top)
        val r = minOf(a.right, b.right)
        val bt = minOf(a.bottom, b.bottom)
        if (r <= l || bt <= t) return 0f
        val inter = (r - l) * (bt - t)
        val areaA = (a.right - a.left) * (a.bottom - a.top)
        val areaB = (b.right - b.left) * (b.bottom - b.top)
        val union = areaA + areaB - inter
        return if (union > 0f) inter / union else 0f
    }

    fun sampleCount(): Int {
        val root = File(context.filesDir, "training_data")
        return root.listFiles()?.count { it.isDirectory } ?: 0
    }

    fun exportZipToDownloads(): String? {
        flushPendingWrites()
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
            val stream = context.contentResolver.openOutputStream(uri)
                ?: throw IOException("Could not open output stream for $uri")
            stream.use { os ->
                ZipOutputStream(os.buffered()).use { zip ->
                    sampleDirs.forEach { dir ->
                        val ts = dir.name
                        val jpg = File(dir, "$ts.jpg")
                        val txt = File(dir, "$ts.txt")
                        if (jpg.exists() && txt.exists()) {
                            zip.putNextEntry(ZipEntry("images/$ts.jpg"))
                            jpg.inputStream().use { it.copyTo(zip) }
                            zip.closeEntry()

                            zip.putNextEntry(ZipEntry("labels/$ts.txt"))
                            txt.inputStream().use { it.copyTo(zip) }
                            zip.closeEntry()
                        }
                    }
                }
            }
            Log.i(TAG, "Exported ${sampleDirs.size} training samples to Downloads/$zipName")
            zipName
        } catch (e: Exception) {
            Log.e(TAG, "Training export failed: ${e.localizedMessage}", e)
            try { context.contentResolver.delete(uri, null, null) } catch (_: Exception) {}
            null
        }
    }

    fun clearAll() {
        flushPendingWrites()
        File(context.filesDir, "training_data").deleteRecursively()
        lastSavedAtMs = 0L
        lastSavedBox = null
        Log.i(TAG, "All collected training samples deleted.")
    }

    private fun flushPendingWrites() {
        try {
            ioExecutor.submit {}.get(3, TimeUnit.SECONDS)
        } catch (_: Exception) {}
    }

    companion object {
        private const val TAG = "TrainingDataCollector"
        private const val MIN_SAMPLE_INTERVAL_MS = 1_500L
        private const val DEDUP_WINDOW_MS = 10_000L
        private const val DEDUP_IOU_THRESHOLD = 0.65f
    }
}
