package com.ct3d.jolt.camera

import android.content.Context
import android.graphics.*
import android.util.Log
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

/**
 * Immutable plate box in fractional (0..1) frame coordinates, drawn by the UI overlay.
 *
 * Replaces mutable android.graphics.RectF in UI/pipeline state: RectF is mutable, so Compose
 * treats state containing it as unstable and recomposes the Main screen every frame. A data class
 * of vals (all primitives) is inferred stable, so the overlay only recomposes when boxes change.
 */
data class PlateBox(val left: Float, val top: Float, val right: Float, val bottom: Float)

/**
 * CameraX ImageAnalysis.Analyzer running at 1 FPS.
 *
 * Pipeline (Phase 2):
 *  1. YOLOv8 license-plate detection on the full frame (A6: ROI crop removed — see below)
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
        // The UI draws green overlay rectangles from these. Immutable PlateBox (not RectF) keeps
        // this state Compose-stable so the Main screen doesn't recompose every frame.
        val plateBoxes: List<PlateBox> = emptyList(),
        // Most recent successful plate crop bitmap (upscaled). Null when no plate in view.
        val lastPlateCrop: Bitmap? = null
    )

    private val _pipelineState = MutableStateFlow(PipelineState())
    val pipelineState: StateFlow<PipelineState> = _pipelineState

    private var lastAnalyzedTimestamp = 0L
    private val frameDispatcher = Dispatchers.Default

    // A15: multi-frame OCR voting. A single ML Kit misread (LT2379 -> LT2879) must not immediately
    // become the active plate and fire a false known-bad-driver alert. We keep a rolling window of
    // the last VOTE_WINDOW token reads and only publish a token once it has appeared at least
    // VOTE_MIN_COUNT times. Guarded by voteLock because reads happen on the ML Kit callback thread
    // while resets happen on the pipeline dispatcher.
    private val voteLock = Any()
    private val recentTokens = ArrayDeque<String>()

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

    override fun analyze(image: ImageProxy) {
        val currentMillis = System.currentTimeMillis()

        // Analysis-rate gate — release frame immediately if too soon (see ANALYSIS_INTERVAL_MS)
        if (currentMillis - lastAnalyzedTimestamp < ANALYSIS_INTERVAL_MS) {
            image.close()
            return
        }
        lastAnalyzedTimestamp = currentMillis

        // Read ImageProxy metadata synchronously before closing, then close immediately.
        // CameraX requires ImageProxy to be closed on the analyzer thread before returning.
        val rotationDegrees = image.imageInfo.rotationDegrees
        val bitmap = try {
            val raw = image.toRgbaBitmap()
            image.close()
            if (rotationDegrees != 0) {
                val rotated = raw.rotate(rotationDegrees.toFloat())
                raw.recycle() // A4(a): pre-rotation frame is dead once rotate() copies it
                rotated
            } else {
                raw
            }
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
        // A6 decision: use the FULL frame — do not crop to a bottom ROI.
        // The old bottom-65% ROI was a wide-angle-dashcam optimization to drop sky/buildings.
        // With the 5× telephoto's tight FOV aimed at same-direction traffic, target plates
        // routinely sit high in the frame (distant vehicles ahead), so cropping the top would cut
        // them off — bad for detection rate (priority #2). After A1–A3, full-frame inference is
        // fast enough that ROI's ~2.1× area saving is unnecessary. cropRoi() deleted as dead code.
        val roiBitmap = frameBitmap
        try {
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
                    // A19: collect high-confidence frames for training (callback no-ops if training
                    // mode is off). Synchronous — completes before frameBitmap is recycled.
                    if (bestPlate.confidence >= TRAINING_MIN_CONFIDENCE) {
                        trainingCallback?.invoke(roiBitmap, bestPlate.box)
                    }

                    // Two independent upscaled copies: one owned by OCR (recycled in the ML Kit
                    // callback), one persisted for FLAG-save. Keeping them separate means recycling
                    // one never pulls the pixels out from under the other.
                    val ocrCrop = upscaleCrop(plateCrop)
                    val persistCrop = upscaleCrop(plateCrop)
                    plateCrop.recycle() // A4(c): raw crop is consumed once both upscales are made

                    // Publish the persisted crop and recycle the one it replaces. Its other owner is
                    // the ViewModel, which only reads it synchronously on FLAG press (>=1 frame apart).
                    val previousCrop = _pipelineState.value.lastPlateCrop
                    _pipelineState.value = _pipelineState.value.copy(lastPlateCrop = persistCrop)
                    if (previousCrop != null && previousCrop !== persistCrop && !previousCrop.isRecycled) {
                        previousCrop.recycle()
                    }

                    runMlKitOcr(ocrCrop) // takes ownership of ocrCrop
                } else {
                    Log.w(TAG, "Plate crop null — clearing display.")
                    clearDetectionState()
                }
            } else {
                Log.d(TAG, "No plate detected — clearing display (Option A).")
                resetVoteBuffer() // plate left frame — start voting fresh for the next one
                clearDetectionState()
            }
        } finally {
            // A4(b): the full frame is fully consumed here — YOLO copies it into its own scratch
            // bitmap, crops are taken synchronously — so recycle it every path to stop frames
            // piling up over multi-hour drives. (roiBitmap === frameBitmap while ROI crop is off.)
            if (!frameBitmap.isRecycled) frameBitmap.recycle()
        }
    }

    /**
     * Upscale a plate crop to ML Kit's ~128px minimum, ALWAYS returning a new bitmap the caller
     * owns — so the source crop can be recycled without affecting the result, and the OCR copy and
     * the persisted copy are independent bitmaps.
     */
    private fun upscaleCrop(src: Bitmap, minDim: Int = 128): Bitmap {
        return if (src.width < minDim || src.height < minDim) {
            val scale = maxOf(minDim.toFloat() / src.width, minDim.toFloat() / src.height)
            val newW = (src.width * scale).toInt().coerceAtLeast(minDim)
            val newH = (src.height * scale).toInt().coerceAtLeast(minDim)
            Log.d(TAG, "Upscaling plate crop ${src.width}×${src.height} → ${newW}×${newH}")
            Bitmap.createScaledBitmap(src, newW, newH, true)
        } else {
            src.copy(src.config ?: Bitmap.Config.ARGB_8888, false)
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

    /**
     * A15: add a token to the rolling vote window and return the consensus token — the most
     * frequent token in the window, but only once it has appeared at least VOTE_MIN_COUNT times.
     * Returns null while no token has enough agreement yet.
     */
    private fun registerAndVote(token: String): String? = synchronized(voteLock) {
        recentTokens.addLast(token)
        while (recentTokens.size > VOTE_WINDOW) recentTokens.removeFirst()
        // Consensus = the most-voted token; ties are broken toward the more plate-plausible token
        // (A17 scoring boost — vote count still dominates, nothing is hard-rejected).
        val best = recentTokens.groupingBy { it }.eachCount().entries
            .maxWithOrNull(compareBy({ it.value }, { plateFormatScore(it.key) }))
            ?: return@synchronized null
        if (best.value >= VOTE_MIN_COUNT) best.key else null
    }

    /**
     * A17: loose US-plate plausibility, used only as a voting tiebreaker (never a hard filter).
     * US formats vary wildly by state, so this stays permissive: it just declines to boost a
     * token that's a run of digits with a single stray letter (e.g. "1234567X"), which is a rare
     * real plate and a common OCR artifact. Returns 1 for plausible, 0 otherwise.
     */
    private fun plateFormatScore(token: String): Int {
        val letters = token.count { it.isLetter() }
        val digits  = token.count { it.isDigit() }
        if (token.length !in 5..8 || letters == 0 || digits == 0) return 0
        if (letters == 1 && digits >= 6) return 0
        return 1
    }

    private fun resetVoteBuffer() = synchronized(voteLock) { recentTokens.clear() }

    /** Map detection boxes from frame pixel space to fractions (0.0–1.0) for the UI overlay. */
    private fun computeFractionalBoxes(detections: List<Detection>, roiBitmap: Bitmap): List<PlateBox> {
        val roiW = roiBitmap.width.toFloat()
        val roiH = roiBitmap.height.toFloat()
        return detections.map { det ->
            PlateBox(
                det.box.left   / roiW,
                det.box.top    / roiH,
                det.box.right  / roiW,
                det.box.bottom / roiH
            )
        }
    }

    /**
     * Runs ML Kit OCR on an already-upscaled plate crop. Takes ownership of [plateBitmap]: ML Kit
     * reads it asynchronously, so it must not be recycled until the task completes — the
     * completion listener recycles it (covers both success and failure).
     */
    private fun runMlKitOcr(plateBitmap: Bitmap) {
        val inputImage = InputImage.fromBitmap(plateBitmap, 0)
        ocrRecognizer.process(inputImage)
            .addOnCompleteListener {
                if (!plateBitmap.isRecycled) plateBitmap.recycle()
            }
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
                        // A15: only publish once the token wins a vote across recent frames.
                        val consensus = registerAndVote(plateToken)
                        if (consensus != null) {
                            Log.i(TAG, "OCR consensus: '$consensus'  (this read: '$plateToken', full: '$clean')")
                            _pipelineState.value = _pipelineState.value.copy(
                                activePlateocr    = consensus,
                                activeVehicleMmc  = null,
                                lastStatusMessage = "OCR: $consensus"
                            )
                        } else {
                            // Accepted into the vote window but not yet confirmed — don't touch
                            // activePlateocr (prevents a single misread firing a known-bad alert).
                            Log.d(TAG, "OCR pending consensus: '$plateToken' (full: '$clean')")
                            _pipelineState.value = _pipelineState.value.copy(
                                lastStatusMessage = "OCR (confirming): $plateToken"
                            )
                        }
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

    /**
     * Convert an RGBA_8888 ImageProxy directly to a Bitmap.
     *
     * Requires ImageAnalysis to be configured with OUTPUT_IMAGE_FORMAT_RGBA_8888, which delivers
     * a single interleaved RGBA plane. We copy that plane straight into an ARGB_8888 Bitmap —
     * no NV21 interleave, no JPEG encode/decode round-trip (which cost ~10–20ms and several large
     * allocations per frame on the old YUV path).
     *
     * The plane's rowStride may include right-edge padding; when it does we build a slightly wider
     * bitmap then crop back to the real width (the padded temp is recycled since it never escapes).
     */
    private fun ImageProxy.toRgbaBitmap(): Bitmap {
        val plane = planes[0]
        val buffer = plane.buffer.apply { rewind() }
        val pixelStride = plane.pixelStride
        val rowStride = plane.rowStride
        val rowPadding = rowStride - pixelStride * width
        val paddedWidth = width + rowPadding / pixelStride

        val padded = Bitmap.createBitmap(paddedWidth, height, Bitmap.Config.ARGB_8888)
        padded.copyPixelsFromBuffer(buffer)

        return if (rowPadding == 0) {
            padded
        } else {
            val cropped = Bitmap.createBitmap(padded, 0, 0, width, height)
            padded.recycle()
            cropped
        }
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

        /**
         * Minimum gap between analyzed frames. 1000ms (1fps) was chosen for the old CPU
         * pipeline's thermal safety; after A1–A4 (GPU delegate, direct RGBA conversion, buffer
         * reuse, bitmap recycling) per-frame cost dropped several-fold, so we analyze faster to
         * catch plates at highway closing speeds (a plate may be readable for well under a second).
         *
         * 500ms = 2fps. Drop toward 333ms (3fps) after the Phase 7 thermal test confirms the
         * device stays below throttling at the mounted operating temperature. Higher rates also
         * feed A15's multi-frame OCR voting more samples per pass.
         */
        const val ANALYSIS_INTERVAL_MS = 500L

        // A15 OCR voting: publish a token only after it appears >=VOTE_MIN_COUNT times within the
        // last VOTE_WINDOW reads. Higher analysis rates (A5) fill this window faster.
        private const val VOTE_WINDOW = 5
        private const val VOTE_MIN_COUNT = 2

        // A19: only collect training frames from confident detections (cleaner labels).
        private const val TRAINING_MIN_CONFIDENCE = 0.7f
    }
}
