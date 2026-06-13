package com.ct3d.jolt.camera

import android.content.Context
import android.graphics.*
import android.media.Image
import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.ct3d.jolt.ml.YoloV8Detector
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream

/**
 * CameraX ImageAnalysis.Analyzer running at 1 FPS.
 *
 * Pipeline (Phase 2):
 *  1. YOLOv8 license-plate detection on the bottom 65% ROI of the frame
 *  2a. Plate found → ML Kit Text Recognition V2 → validate US plate regex → emit OCR string
 *  2b. No plate    → clear display (Option A: chip disappears)
 *
 * Phase 5 will add the MMC fallback (EfficientNet-Lite0 make/model/color classifier).
 * Phase 6 adds the trainingCallback for annotated frame collection.
 */
class TelephotoAnalyzer(
    private val context: Context,
    private val uiScope: CoroutineScope
) : ImageAnalysis.Analyzer {

    data class PipelineState(
        val activePlateocr: String? = null,
        val activeVehicleMmc: String? = null,
        val fpsTimestamp: Long = 0L,
        val processingTimeMs: Long = 0,
        val lastStatusMessage: String = "Pipeline Idle",
        // Plate boxes as fractions of the full camera frame (0.0–1.0).
        // The UI draws green overlay rectangles from these.
        val plateBoxes: List<RectF> = emptyList(),
        // Most recent successful plate crop bitmap (upscaled). Null when no plate in view.
        val lastPlateCrop: Bitmap? = null
    )

    private val _pipelineState = MutableStateFlow(PipelineState())
    val pipelineState: StateFlow<PipelineState> = _pipelineState

    private var lastAnalyzedTimestamp = 0L
    private val frameDispatcher = Dispatchers.Default

    // Phase 6: set this callback to receive (fullFrameBitmap, plateBox) for training data collection
    var trainingCallback: ((Bitmap, RectF) -> Unit)? = null

    // Lazy ML Kit Latin OCR client (bundled — works fully offline)
    private val ocrRecognizer by lazy {
        TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    }

    // YOLOv8 TFLite detector for vehicle and license plate detection
    private val yoloDetector by lazy {
        YoloV8Detector(context).apply {
            try {
                initialize()
                Log.i(TAG, "YOLOv8 detector initialized successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize YOLOv8 detector: ${e.message}", e)
            }
        }
    }

    data class Detection(
        val box: RectF,
        val label: String,
        val confidence: Float
    )

    @OptIn(ExperimentalGetImage::class)
    override fun analyze(image: ImageProxy) {
        val currentMillis = System.currentTimeMillis()

        // 1 FPS gate — release frame immediately if too soon
        if (currentMillis - lastAnalyzedTimestamp < 1000L) {
            image.close()
            return
        }
        lastAnalyzedTimestamp = currentMillis

        val mediaImage = image.image ?: run {
            Log.w(TAG, "Frame skipped: null media image payload.")
            image.close()
            return
        }

        // Read ImageProxy metadata synchronously before closing, then close immediately.
        // CameraX requires ImageProxy to be closed on the analyzer thread before returning.
        val rotationDegrees = image.imageInfo.rotationDegrees
        val bitmap = try {
            val raw = mediaImage.toBitmap()
            image.close()
            if (rotationDegrees != 0) raw.rotate(rotationDegrees.toFloat()) else raw
        } catch (e: Exception) {
            image.close()
            Log.e(TAG, "Frame decode failed: ${e.localizedMessage}", e)
            return
        }

        uiScope.launch(frameDispatcher) {
            val startTime = System.currentTimeMillis()
            try {
                _pipelineState.value = _pipelineState.value.copy(
                    fpsTimestamp = currentMillis,
                    lastStatusMessage = "Analyzing frame..."
                )

                executeHybridMlPipeline(bitmap)

                val elapsed = System.currentTimeMillis() - startTime
                Log.d(TAG, "Pipeline completed in ${elapsed}ms")
                _pipelineState.value = _pipelineState.value.copy(processingTimeMs = elapsed)

            } catch (e: Exception) {
                Log.e(TAG, "Pipeline exception: ${e.localizedMessage}", e)
                _pipelineState.value = _pipelineState.value.copy(
                    lastStatusMessage = "Pipeline error: ${e.localizedMessage}"
                )
            }
        }
    }

    /**
     * Phase 2 pipeline — single-class license plate detection.
     * 1. Crop bottom 65% (discard sky / trees / buildings).
     * 2. Run YOLOv8 → find best plate box.
     * 3a. Plate found → compute fractional overlay boxes → crop → ML Kit OCR.
     * 3b. No plate    → clear display (Option A).
     */
    private suspend fun executeHybridMlPipeline(frameBitmap: Bitmap) {
        // Use full frame — ROI crop removed so plates anywhere in frame are detected.
        // The 35%-top crop was a dashcam optimization for sky removal; re-enable in Phase 7
        // once we confirm the model works reliably in vehicle deployment.
        val roiBitmap = frameBitmap

        _pipelineState.value = _pipelineState.value.copy(lastStatusMessage = "Scanning...")
        val detections = runYoloDetection(roiBitmap)
        Log.d(TAG, "YOLO: ${detections.size} plate detections (threshold 0.25)")

        val bestPlate = detections.filter { it.confidence > 0.25f }
                                  .maxByOrNull { it.confidence }

        if (bestPlate != null) {
            val fractionalBoxes = computeFractionalBoxes(
                detections.filter { it.confidence > 0.25f }, roiBitmap
            )
            _pipelineState.value = _pipelineState.value.copy(
                plateBoxes = fractionalBoxes,
                lastStatusMessage = "Plate found — running OCR..."
            )

            Log.d(TAG, "Best plate box: left=${bestPlate.box.left} top=${bestPlate.box.top} " +
                       "right=${bestPlate.box.right} bottom=${bestPlate.box.bottom} " +
                       "conf=${bestPlate.confidence}  ROI=${roiBitmap.width}×${roiBitmap.height}")
            val plateCrop = cropBitmapToDetection(roiBitmap, bestPlate.box)
            if (plateCrop != null) {
                trainingCallback?.invoke(roiBitmap, bestPlate.box) // Phase 6 hook
                runMlKitOcr(plateCrop)
                // Persist upscaled crop in pipeline state so ViewModel can save it on FLAG press
                val minDim = 128
                val scaled = if (plateCrop.width < minDim || plateCrop.height < minDim) {
                    val scale = maxOf(minDim.toFloat() / plateCrop.width, minDim.toFloat() / plateCrop.height)
                    Bitmap.createScaledBitmap(plateCrop, (plateCrop.width * scale).toInt().coerceAtLeast(minDim), (plateCrop.height * scale).toInt().coerceAtLeast(minDim), true)
                } else plateCrop
                _pipelineState.value = _pipelineState.value.copy(lastPlateCrop = scaled)
            } else {
                Log.w(TAG, "Plate crop null — clearing display.")
                clearDetectionState()
            }
        } else {
            Log.d(TAG, "No plate detected — clearing display (Option A).")
            clearDetectionState()
        }
    }

    private fun clearDetectionState() {
        _pipelineState.value = _pipelineState.value.copy(
            activePlateocr   = null,
            activeVehicleMmc = null,
            plateBoxes       = emptyList(),
            lastStatusMessage = "Scanning..."
        )
    }

    /** Map detection boxes from frame pixel space to fractions (0.0–1.0) for the UI overlay. */
    private fun computeFractionalBoxes(detections: List<Detection>, roiBitmap: Bitmap): List<RectF> {
        val roiW = roiBitmap.width.toFloat()
        val roiH = roiBitmap.height.toFloat()
        return detections.map { det ->
            RectF(
                det.box.left   / roiW,
                det.box.top    / roiH,
                det.box.right  / roiW,
                det.box.bottom / roiH
            )
        }
    }

    /** Crop to lower 65% of the frame — removes sky, trees, buildings from inference. */
    private fun cropRoi(bitmap: Bitmap): Bitmap {
        val topOffset = (bitmap.height * 0.35f).toInt()
        val roiHeight = bitmap.height - topOffset
        return if (roiHeight > 0) {
            Bitmap.createBitmap(bitmap, 0, topOffset, bitmap.width, roiHeight)
        } else {
            bitmap
        }
    }

    private fun runMlKitOcr(plateBitmap: Bitmap) {
        // ML Kit requires minimum 32×32; upscale small crops so OCR has enough pixels to work with.
        val minDim = 128
        val scaledBitmap = if (plateBitmap.width < minDim || plateBitmap.height < minDim) {
            val scale = maxOf(minDim.toFloat() / plateBitmap.width, minDim.toFloat() / plateBitmap.height)
            val newW = (plateBitmap.width * scale).toInt().coerceAtLeast(minDim)
            val newH = (plateBitmap.height * scale).toInt().coerceAtLeast(minDim)
            Log.d(TAG, "Upscaling plate crop ${plateBitmap.width}×${plateBitmap.height} → ${newW}×${newH}")
            Bitmap.createScaledBitmap(plateBitmap, newW, newH, true)
        } else {
            plateBitmap
        }
        val inputImage = InputImage.fromBitmap(scaledBitmap, 0)
        ocrRecognizer.process(inputImage)
            .addOnSuccessListener { visionText ->
                val raw = visionText.text.trim().replace("\n", " ")
                if (raw.isNotEmpty()) {
                    val clean = raw.uppercase().filter { it.isLetterOrDigit() || it == ' ' }
                    // US plates are 5–8 alphanumeric chars. ML Kit reads the full plate face
                    // including state name ("CONNECTICUT LT2379 CONSTITUTION STATE"), so we
                    // scan each space-delimited token for the one that looks like a plate number:
                    // 5–8 chars, contains at least one digit and at least one letter.
                    val plateToken = clean.split(" ").firstOrNull { token ->
                        token.length in 5..8 &&
                        token.any { it.isDigit() } &&
                        token.any { it.isLetter() }
                    }
                    if (plateToken != null) {
                        Log.i(TAG, "OCR success: '$plateToken'  (full read: '$clean')")
                        _pipelineState.value = _pipelineState.value.copy(
                            activePlateocr    = plateToken,
                            activeVehicleMmc  = null,
                            lastStatusMessage = "OCR: $plateToken"
                        )
                    } else {
                        Log.w(TAG, "OCR '$clean' — no plate-like token found, clearing display.")
                        clearDetectionState()
                    }
                } else {
                    Log.w(TAG, "ML Kit returned empty — clearing display.")
                    clearDetectionState()
                }
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "ML Kit OCR failed: ${e.localizedMessage}", e)
                clearDetectionState()
            }
    }

    /**
     * Phase 5 stub — EfficientNet-Lite0 make/model/color classifier.
     * Not called in Phase 2 (Option A: show nothing when no plate detected).
     */
    @Suppress("unused")
    private fun runMmcClassification(vehicleBitmap: Bitmap): String {
        val makes  = listOf("Tesla Model Y", "Toyota RAV4", "Honda Accord", "Ford F-150", "Chevy Silverado")
        val colors = listOf("White", "Charcoal", "Silver", "Red", "Blue")
        val m = makes[(vehicleBitmap.width + vehicleBitmap.height) % makes.size]
        val c = colors[(vehicleBitmap.width * vehicleBitmap.height) % colors.size]
        return "$c $m"
    }

    /**
     * YOLOv8 Object Detection — REAL IMPLEMENTATION
     * Runs TFLite model with GPU acceleration for license plate and vehicle detection.
     */
    private fun runYoloDetection(bitmap: Bitmap): List<Detection> {
        return try {
            val yoloResults = yoloDetector.detect(bitmap, confidenceThreshold = 0.25f)

            // Convert YoloV8Detector.Detection to TelephotoAnalyzer.Detection
            yoloResults.map { detection ->
                Detection(
                    box = detection.box,
                    label = detection.label,
                    confidence = detection.confidence
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "YOLO detection failed: ${e.message}", e)
            emptyList()
        }
    }

    private fun cropBitmapToDetection(src: Bitmap, box: RectF): Bitmap? {
        return try {
            val l = box.left.coerceIn(0f, src.width.toFloat()).toInt()
            val t = box.top.coerceIn(0f, src.height.toFloat()).toInt()
            val r = box.right.coerceIn(0f, src.width.toFloat()).toInt()
            val b = box.bottom.coerceIn(0f, src.height.toFloat()).toInt()
            val w = r - l; val h = b - t
            if (w > 0 && h > 0) Bitmap.createBitmap(src, l, t, w, h) else null
        } catch (e: Exception) { null }
    }

    // YUV_420_888 → NV21 conversion. The U and V planes in YUV_420_888 are NOT guaranteed
    // to be contiguous, so we must interleave V and U bytes manually to produce valid NV21
    // (which is Y-plane followed by interleaved VU pairs). Concatenating the raw plane buffers
    // produces corrupt green frames on most Pixel devices.
    private fun Image.toBitmap(): Bitmap {
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]

        val yBuf = yPlane.buffer
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer

        val ySize = yBuf.remaining()
        // Chroma planes are width/2 × height/2 — total chroma bytes = width*height/2
        val chromaSize = width * height / 2

        val nv21 = ByteArray(ySize + chromaSize)
        yBuf.get(nv21, 0, ySize)

        // Interleave V and U into NV21 VU pairs. pixelStride is typically 2 for the
        // interleaved case (both U and V buffers point into the same underlying memory),
        // but we handle pixelStride == 1 (planar) as well.
        val vPixelStride = vPlane.pixelStride
        val uPixelStride = uPlane.pixelStride
        val vRowStride = vPlane.rowStride
        val uRowStride = uPlane.rowStride
        val vBytes = ByteArray(vBuf.remaining()).also { vBuf.get(it) }
        val uBytes = ByteArray(uBuf.remaining()).also { uBuf.get(it) }

        var dstIdx = ySize
        val chromaHeight = height / 2
        val chromaWidth  = width  / 2
        for (row in 0 until chromaHeight) {
            for (col in 0 until chromaWidth) {
                val vIdx = row * vRowStride + col * vPixelStride
                val uIdx = row * uRowStride + col * uPixelStride
                nv21[dstIdx++] = vBytes[vIdx]
                nv21[dstIdx++] = uBytes[uIdx]
            }
        }

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, this.width, this.height), 95, out)
        return BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
    }

    private fun Bitmap.rotate(degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
    }

    /**
     * Clean up resources when analyzer is no longer needed.
     * Call this when stopping the camera or closing the app.
     */
    fun cleanup() {
        try {
            yoloDetector.close()
            Log.i(TAG, "TelephotoAnalyzer cleaned up successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error during cleanup: ${e.message}", e)
        }
    }

    companion object {
        private const val TAG = "TelephotoAnalyzer"
    }
}
