package com.ct3d.jolt.camera

import android.content.Context
import android.graphics.*
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.ct3d.jolt.data.StructuredAppearance
import com.ct3d.jolt.ml.MmcClassifier
import com.ct3d.jolt.ml.YoloV8Detector
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.coroutines.resume
import kotlin.math.sqrt

/**
 * Immutable plate box in fractional (0..1) frame coordinates, drawn by the UI overlay.
 */
data class PlateBox(val left: Float, val top: Float, val right: Float, val bottom: Float)

/**
 * CameraX ImageAnalysis.Analyzer running at 2 FPS (`ANALYSIS_INTERVAL_MS = 500L`).
 *
 * Phase 2.1 + Phase 3 Upgrades:
 *  - Latency Trim: skips ML Kit OCR on tiny distant boxes (`width < MIN_OCR_BOX_WIDTH_PX = 26px`)
 *    where characters are <6px tall and unreadable, prioritizing foreground boxes (`MAX_OCR = 2`)
 *    while still rendering all detected green boxes on the HUD.
 *  - Structured Appearance (`Color + Make + BodyStyle`): publishes marginalized vehicle appearance
 *    for Analytics encounter denominators and Tier 2/3 Partial-Plate Matching.
 *  - Partial Plate Fragment Hint: extracts 3–4 char alphanumeric fragments when a tinted/angled
 *    plate cannot yield a full 5–8 char consensus token.
 *
 * Threading (R1): ALL TFLite GPU delegate init and inference stay strictly on `jolt-pipeline`.
 */
class TelephotoAnalyzer(
    private val context: Context,
    private val uiScope: CoroutineScope
) : ImageAnalysis.Analyzer {

    data class PipelineState(
        val activePlateocr: String? = null,
        val activeVehicleMmc: String? = null,
        val activeAppearance: StructuredAppearance? = null,
        val partialPlateHint: String? = null,
        val confirmedPlateTokens: List<String> = emptyList(),
        val fpsTimestamp: Long = 0L,
        val processingTimeMs: Long = 0,
        val lastStatusMessage: String = "Pipeline Idle",
        val plateBoxes: List<PlateBox> = emptyList(),
        val frameWidth: Int = 0,
        val frameHeight: Int = 0,
        val lastPlateCrop: Bitmap? = null
    )

    private val _pipelineState = MutableStateFlow(PipelineState())
    val pipelineState: StateFlow<PipelineState> = _pipelineState

    private var lastAnalyzedTimestamp = 0L

    // R1: single dedicated pipeline thread for TFLite GPU EGL context affinity.
    private val pipelineExecutor = Executors.newSingleThreadExecutor { r -> Thread(r, "jolt-pipeline") }
    private val frameDispatcher = pipelineExecutor.asCoroutineDispatcher()

    // R2: drop frames arriving while previous frame is still processing.
    private val pipelineBusy = AtomicBoolean(false)
    private val busyDropsSinceLastFrame = AtomicInteger(0)
    private var previousExecutedFrameAtMs = 0L

    private val voteBuffer = OcrVoteBuffer(VOTE_WINDOW, VOTE_MIN_COUNT)

    var trainingCallback: ((Bitmap, List<RectF>) -> Unit)? = null
    var frameTelemetryCallback: ((FrameTelemetry) -> Unit)? = null

    data class FrameTelemetry(
        val timestampMs: Long,
        val frameDeltaMs: Long,
        val busyDrops: Int,
        val totalMs: Long,
        val yoloMs: Long,
        val ocrMs: Long,
        val mmcMs: Long,
        val yoloGpu: Boolean,
        val mmcGpu: Boolean,
        val plateBoxCount: Int,
        val topPlateConf: Float,
        val ocrTokens: List<String>,
        val consensusPlate: String?,
        val mmcTrigger: String,
        val mmcPublished: String?,
        val mmcTop1Conf: Float,
        val mmcTop3: String?
    )

    private val ocrRecognizer by lazy {
        TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    }

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

    private val mmcClassifier by lazy {
        MmcClassifier(context).apply { initialize() }
    }

    data class Detection(
        val box: RectF,
        val label: String,
        val confidence: Float
    )

    override fun analyze(image: ImageProxy) {
        val currentMillis = System.currentTimeMillis()
        if (currentMillis - lastAnalyzedTimestamp < ANALYSIS_INTERVAL_MS) {
            image.close()
            return
        }
        lastAnalyzedTimestamp = currentMillis

        if (!pipelineBusy.compareAndSet(false, true)) {
            busyDropsSinceLastFrame.incrementAndGet()
            image.close()
            return
        }

        val rotationDegrees = image.imageInfo.rotationDegrees
        val bitmap = try {
            val raw = image.toRgbaBitmap()
            image.close()
            if (rotationDegrees != 0) {
                val rotated = raw.rotate(rotationDegrees.toFloat())
                raw.recycle()
                rotated
            } else {
                raw
            }
        } catch (e: Exception) {
            image.close()
            pipelineBusy.set(false)
            Log.e(TAG, "Frame decode failed: ${e.localizedMessage}", e)
            return
        }

        uiScope.launch(frameDispatcher) {
            val startTime = System.currentTimeMillis()
            val frameDelta = if (previousExecutedFrameAtMs > 0L) startTime - previousExecutedFrameAtMs else 0L
            previousExecutedFrameAtMs = startTime
            val droppedWhileBusy = busyDropsSinceLastFrame.getAndSet(0)
            try {
                _pipelineState.value = _pipelineState.value.copy(
                    fpsTimestamp = currentMillis,
                    lastStatusMessage = "Analyzing frame..."
                )

                val metrics = executeHybridMlPipeline(bitmap)

                val elapsed = System.currentTimeMillis() - startTime
                Log.d(TAG, "Pipeline completed in ${elapsed}ms")
                _pipelineState.value = _pipelineState.value.copy(processingTimeMs = elapsed)

                frameTelemetryCallback?.invoke(
                    FrameTelemetry(
                        timestampMs    = startTime,
                        frameDeltaMs   = frameDelta,
                        busyDrops      = droppedWhileBusy,
                        totalMs        = elapsed,
                        yoloMs         = metrics.yoloMs,
                        ocrMs          = metrics.ocrMs,
                        mmcMs          = metrics.mmcMs,
                        yoloGpu        = yoloDetector.isUsingGpuDelegate,
                        mmcGpu         = mmcClassifier.isUsingGpuDelegate,
                        plateBoxCount  = metrics.plateBoxCount,
                        topPlateConf   = metrics.topPlateConf,
                        ocrTokens      = metrics.ocrTokens,
                        consensusPlate = metrics.consensusPlate,
                        mmcTrigger     = metrics.mmcTrigger,
                        mmcPublished   = metrics.mmcPublished,
                        mmcTop1Conf    = metrics.mmcTop1Conf,
                        mmcTop3        = metrics.mmcTop3
                    )
                )
            } catch (e: Exception) {
                Log.e(TAG, "Pipeline exception: ${e.localizedMessage}", e)
                _pipelineState.value = _pipelineState.value.copy(
                    lastStatusMessage = "Pipeline error: ${e.localizedMessage}"
                )
            } finally {
                pipelineBusy.set(false)
            }
        }
    }

    private data class PipelineMetrics(
        val yoloMs: Long = 0L,
        val ocrMs: Long = 0L,
        val mmcMs: Long = 0L,
        val plateBoxCount: Int = 0,
        val topPlateConf: Float = 0f,
        val ocrTokens: List<String> = emptyList(),
        val consensusPlate: String? = null,
        val mmcTrigger: String = "NONE",
        val mmcPublished: String? = null,
        val mmcTop1Conf: Float = 0f,
        val mmcTop3: String? = null
    )

    private suspend fun executeHybridMlPipeline(frameBitmap: Bitmap): PipelineMetrics {
        val roiBitmap = frameBitmap
        data class PlateCandidate(val det: Detection, val ocrCrop: Bitmap, val persistCrop: Bitmap)
        val candidates = mutableListOf<PlateCandidate>()
        var mmcFallback: Bitmap? = null
        val yoloElapsedMs: Long
        var mmcElapsedMs = 0L
        var mmcTrigger = "NONE"
        var mmcDetail: MmcClassifier.DetailedPrediction? = null
        val boxCount: Int
        val topConf: Float

        try {
            _pipelineState.value = _pipelineState.value.copy(
                lastStatusMessage = "Scanning...",
                frameWidth = roiBitmap.width,
                frameHeight = roiBitmap.height
            )
            val yoloStart = System.currentTimeMillis()
            val detections = runYoloDetection(roiBitmap)
            yoloElapsedMs = System.currentTimeMillis() - yoloStart

            val visiblePlates = detections
                .filter { it.confidence > 0.25f }
                .sortedByDescending { det ->
                    val area = maxOf(1f, det.box.width() * det.box.height())
                    det.confidence * sqrt(area)
                }

            boxCount = visiblePlates.size
            topConf = visiblePlates.maxOfOrNull { it.confidence } ?: 0f

            if (visiblePlates.isNotEmpty()) {
                val fractionalBoxes = computeFractionalBoxes(visiblePlates.take(4), roiBitmap)
                _pipelineState.value = _pipelineState.value.copy(
                    plateBoxes = fractionalBoxes,
                    lastStatusMessage = "Plate found — running OCR..."
                )

                val bestPlate = visiblePlates.first()

                // Phase 6: invoke async training saver only when enabled
                val trainingBoxes = detections
                    .filter { it.confidence >= TRAINING_MIN_CONFIDENCE }
                    .map { it.box }
                if (trainingBoxes.isNotEmpty()) {
                    trainingCallback?.invoke(roiBitmap, trainingBoxes)
                }

                // Phase 2.1 Latency Trim: skip ML Kit OCR on tiny far-highway boxes (<26px wide)
                // where characters are <6px tall and guaranteed unreadable, saving ~150-240ms!
                val ocrEligible = visiblePlates
                    .filter { it.box.width() >= MIN_OCR_BOX_WIDTH_PX && it.box.height() >= MIN_OCR_BOX_HEIGHT_PX }
                    .take(MAX_OCR_PLATES_PER_FRAME)

                for (det in ocrEligible) {
                    val rawCrop = cropBitmapToDetection(roiBitmap, det.box) ?: continue
                    val ocrCrop = upscaleCrop(rawCrop)
                    val persistCopy = upscaleCrop(rawCrop)
                    rawCrop.recycle()
                    candidates.add(PlateCandidate(det, ocrCrop, persistCopy))
                }

                if (mmcClassifier.isAvailable) {
                    mmcFallback = buildMmcInput(roiBitmap, bestPlate.box)
                }

                // If boxes exist only far down-road (<26px wide), run anchored MMC right away
                val farVehicleCrop = mmcFallback
                if (candidates.isEmpty() && farVehicleCrop != null) {
                    mmcFallback = null
                    mmcTrigger = "DISTANT_BOX_ANCHORED"
                    val mmcStart = System.currentTimeMillis()
                    mmcDetail = classifyAndPublishMmc(farVehicleCrop)
                    mmcElapsedMs = System.currentTimeMillis() - mmcStart
                }
            } else {
                resetVoteBuffer()
                clearDetectionState()
                if (mmcClassifier.isAvailable) {
                    mmcTrigger = "NO_PLATE_CENTER"
                    val mmcStart = System.currentTimeMillis()
                    mmcDetail = classifyAndPublishMmc(
                        vehicleBitmap = buildMmcInput(roiBitmap),
                        minConfidence = NO_PLATE_MMC_MIN_CONFIDENCE
                    )
                    mmcElapsedMs = System.currentTimeMillis() - mmcStart
                }
            }
        } finally {
            if (!frameBitmap.isRecycled) frameBitmap.recycle()
        }

        if (candidates.isEmpty()) {
            recycleQuietly(mmcFallback)
            return PipelineMetrics(
                yoloMs        = yoloElapsedMs,
                mmcMs         = mmcElapsedMs,
                plateBoxCount = boxCount,
                topPlateConf  = topConf,
                mmcTrigger    = mmcTrigger,
                mmcPublished  = mmcDetail?.published?.label,
                mmcTop1Conf   = mmcDetail?.top1Confidence ?: 0f,
                mmcTop3       = mmcDetail?.top3Summary
            )
        }

        val extracted = mutableListOf<Pair<PlateCandidate, String>>()
        var partialFragment: String? = null
        val ocrStart = System.currentTimeMillis()
        try {
            for (candidate in candidates) {
                val raw = processOcr(candidate.ocrCrop)
                val clean = raw?.uppercase()?.filter { it.isLetterOrDigit() || it == ' ' } ?: ""
                val token = if (clean.isNotEmpty()) extractPlateToken(clean) else null
                if (token != null) {
                    extracted.add(candidate to token)
                } else if (partialFragment == null && clean.isNotEmpty()) {
                    partialFragment = extractPartialPlateFragment(clean)
                }
            }
        } finally {
            for (c in candidates) recycleQuietly(c.ocrCrop)
        }
        val ocrElapsedMs = System.currentTimeMillis() - ocrStart

        if (extracted.isNotEmpty()) {
            // Fast sub-millisecond HSV paint color check on mmcFallback before recycling
            val quickColor = mmcFallback?.let { bmp ->
                val (col, conf) = mmcClassifier.estimateColorOnly(bmp)
                recycleQuietly(bmp)
                if (col != null) StructuredAppearance(color = col, colorConfidence = conf) else null
            }
            mmcFallback = null

            val chosenPersist = extracted.first().first.persistCrop
            for (c in candidates) {
                if (c.persistCrop !== chosenPersist) recycleQuietly(c.persistCrop)
            }
            _pipelineState.value = _pipelineState.value.copy(lastPlateCrop = chosenPersist)

            val tokensInFrame = extracted.map { it.second }
            val voteResult = registerAndVoteFrame(tokensInFrame)
            val consensus = voteResult.primaryConsensus
            if (consensus != null) {
                Log.i(TAG, "OCR consensus: '$consensus' (frame tokens=$tokensInFrame)")
                _pipelineState.value = _pipelineState.value.copy(
                    activePlateocr       = consensus,
                    confirmedPlateTokens = voteResult.confirmedTokens,
                    activeVehicleMmc     = null,
                    activeAppearance     = quickColor,
                    partialPlateHint     = null,
                    lastStatusMessage    = "OCR: $consensus"
                )
            } else {
                val pending = tokensInFrame.first()
                _pipelineState.value = _pipelineState.value.copy(
                    activeVehicleMmc     = null,
                    activeAppearance     = quickColor,
                    partialPlateHint     = pending,
                    confirmedPlateTokens = emptyList(),
                    lastStatusMessage    = "OCR (confirming): $pending"
                )
            }
            return PipelineMetrics(
                yoloMs         = yoloElapsedMs,
                ocrMs          = ocrElapsedMs,
                plateBoxCount  = boxCount,
                topPlateConf   = topConf,
                ocrTokens      = tokensInFrame,
                consensusPlate = consensus
            )
        } else {
            val fallbackPersist = candidates.first().persistCrop
            for (c in candidates) {
                if (c.persistCrop !== fallbackPersist) recycleQuietly(c.persistCrop)
            }
            _pipelineState.value = _pipelineState.value.copy(
                activePlateocr       = null,
                confirmedPlateTokens = emptyList(),
                partialPlateHint     = partialFragment,
                lastPlateCrop        = fallbackPersist,
                lastStatusMessage    = "Plate unreadable — trying MMC fallback..."
            )
            if (mmcFallback != null) {
                mmcTrigger = "OCR_FAIL_ANCHORED"
                val mmcStart = System.currentTimeMillis()
                mmcDetail = classifyAndPublishMmc(mmcFallback, partialFragment)
                mmcElapsedMs = System.currentTimeMillis() - mmcStart
            }
            return PipelineMetrics(
                yoloMs        = yoloElapsedMs,
                ocrMs         = ocrElapsedMs,
                mmcMs         = mmcElapsedMs,
                plateBoxCount = boxCount,
                topPlateConf  = topConf,
                mmcTrigger    = mmcTrigger,
                mmcPublished  = mmcDetail?.published?.label,
                mmcTop1Conf   = mmcDetail?.top1Confidence ?: 0f,
                mmcTop3       = mmcDetail?.top3Summary
            )
        }
    }

    private suspend fun processOcr(plateBitmap: Bitmap): String? =
        suspendCancellableCoroutine { cont ->
            val inputImage = InputImage.fromBitmap(plateBitmap, 0)
            ocrRecognizer.process(inputImage)
                .addOnCompleteListener {
                    recycleQuietly(plateBitmap)
                }
                .addOnSuccessListener { visionText ->
                    val raw = visionText.text.trim().replace("\n", " ").ifEmpty { null }
                    if (cont.isActive) cont.resume(raw)
                }
                .addOnFailureListener { e ->
                    Log.e(TAG, "ML Kit OCR failed: ${e.localizedMessage}", e)
                    if (cont.isActive) cont.resume(null)
                }
        }

    private fun upscaleCrop(src: Bitmap, minDim: Int = 128): Bitmap {
        return if (src.width < minDim || src.height < minDim) {
            val scale = maxOf(minDim.toFloat() / src.width, minDim.toFloat() / src.height)
            val newW = (src.width * scale).toInt().coerceAtLeast(minDim)
            val newH = (src.height * scale).toInt().coerceAtLeast(minDim)
            Bitmap.createScaledBitmap(src, newW, newH, true)
        } else {
            src.copy(src.config ?: Bitmap.Config.ARGB_8888, false)
        }
    }

    private fun clearDetectionState() {
        _pipelineState.value = _pipelineState.value.copy(
            activePlateocr       = null,
            activeVehicleMmc     = null,
            activeAppearance     = null,
            partialPlateHint     = null,
            confirmedPlateTokens = emptyList(),
            plateBoxes           = emptyList(),
            lastPlateCrop        = null, // R4: ViewModel owns published crops
            lastStatusMessage    = "Scanning..."
        )
    }

    private fun registerAndVoteFrame(tokensInFrame: List<String>): VoteResult =
        voteBuffer.registerFrame(tokensInFrame)

    private fun resetVoteBuffer() = voteBuffer.reset()

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

    private fun buildMmcInput(src: Bitmap, plateBox: RectF? = null): Bitmap {
        val fw = src.width.toFloat()
        val fh = src.height.toFloat()

        val crop = if (plateBox != null && plateBox.width() > 1f && plateBox.height() > 1f) {
            val pw = plateBox.width()
            val ph = plateBox.height()
            val cx = plateBox.centerX()
            RectF(
                cx - MMC_CROP_WIDTH_MULT / 2f * pw,
                plateBox.top - MMC_CROP_UP_MULT * ph,
                cx + MMC_CROP_WIDTH_MULT / 2f * pw,
                plateBox.bottom + MMC_CROP_DOWN_MULT * ph
            )
        } else {
            val side = minOf(fw, fh)
            val cx = fw / 2f
            val cy = fh / 2f
            RectF(cx - side / 2f, cy - side / 2f, cx + side / 2f, cy + side / 2f)
        }

        val l = crop.left.coerceIn(0f, fw - 1f).toInt()
        val t = crop.top.coerceIn(0f, fh - 1f).toInt()
        val r = crop.right.coerceIn(l + 1f, fw).toInt()
        val b = crop.bottom.coerceIn(t + 1f, fh).toInt()
        val w = r - l
        val h = b - t

        val cropped = if (w > 1 && h > 1 && (w < src.width || h < src.height)) {
            try {
                Bitmap.createBitmap(src, l, t, w, h)
            } catch (e: Exception) {
                Log.w(TAG, "MMC crop failed (${e.message}) — using full frame")
                null
            }
        } else null

        val source = cropped ?: src
        val scaled = Bitmap.createScaledBitmap(source, MMC_INPUT_SIZE, MMC_INPUT_SIZE, true)
        if (cropped != null && cropped !== scaled && cropped !== src) cropped.recycle()
        return scaled
    }

    private fun classifyAndPublishMmc(
        vehicleBitmap: Bitmap,
        partialFragment: String? = null,
        minConfidence: Float = MmcClassifier.MIN_MMC_CONFIDENCE
    ): MmcClassifier.DetailedPrediction? {
        return try {
            if (_pipelineState.value.activePlateocr != null) {
                return null
            }
            val detail = mmcClassifier.classifyDetailed(vehicleBitmap, minConfidence = minConfidence) ?: return null
            val prediction = detail.published
            if (prediction != null) {
                Log.i(TAG, "MMC fallback: ${prediction.label} (conf=${"%.3f".format(prediction.confidence)})")
                _pipelineState.value = _pipelineState.value.copy(
                    activePlateocr   = null,
                    activeVehicleMmc = prediction.label,
                    activeAppearance = prediction.appearance,
                    partialPlateHint = partialFragment,
                    lastStatusMessage = "MMC: ${prediction.label}"
                )
            } else if (!detail.appearance.isEmpty && minConfidence <= MmcClassifier.MIN_MMC_CONFIDENCE) {
                _pipelineState.value = _pipelineState.value.copy(
                    activeAppearance = detail.appearance,
                    partialPlateHint = partialFragment
                )
            }
            detail
        } catch (e: Exception) {
            Log.e(TAG, "MMC fallback failed: ${e.localizedMessage}", e)
            null
        } finally {
            recycleQuietly(vehicleBitmap)
        }
    }

    private fun recycleQuietly(bmp: Bitmap?) {
        if (bmp != null && !bmp.isRecycled) bmp.recycle()
    }

    private fun runYoloDetection(bitmap: Bitmap): List<Detection> {
        return try {
            yoloDetector.detect(bitmap, confidenceThreshold = 0.25f).map { detection ->
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
            val w = r - l
            val h = b - t
            if (w > 0 && h > 0) Bitmap.createBitmap(src, l, t, w, h) else null
        } catch (_: Exception) {
            null
        }
    }

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

    fun cleanup() {
        try {
            yoloDetector.close()
            mmcClassifier.close()
            Log.i(TAG, "TelephotoAnalyzer cleaned up successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error during cleanup: ${e.message}", e)
        }
        try {
            frameDispatcher.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error closing pipeline dispatcher: ${e.message}", e)
        }
    }

    companion object {
        private const val TAG = "TelephotoAnalyzer"

        const val ANALYSIS_INTERVAL_MS = 500L

        private const val VOTE_WINDOW = 5
        private const val VOTE_MIN_COUNT = 2
        private const val TRAINING_MIN_CONFIDENCE = 0.7f
        private const val MMC_INPUT_SIZE = 224

        private const val MMC_CROP_WIDTH_MULT = 6.0f
        private const val MMC_CROP_UP_MULT    = 4.5f
        private const val MMC_CROP_DOWN_MULT  = 1.5f

        // Phase 2.1 Latency Trim: cap OCR to top 2 foreground plates & skip micro-boxes (<26x11 px)
        private const val MAX_OCR_PLATES_PER_FRAME = 2
        private const val MIN_OCR_BOX_WIDTH_PX = 26f
        private const val MIN_OCR_BOX_HEIGHT_PX = 11f
        // Higher floor when no plate box anchors the scene so empty road/desk textures don't false-fire
        private const val NO_PLATE_MMC_MIN_CONFIDENCE = 0.64f

        private val STATE_WORDS = setOf(
            "CONNECTICUT", "CONSTITUTION", "STATE", "NEW", "YORK", "MASSACHUSETTS",
            "SPIRIT", "AMERICA", "JERSEY", "GARDEN", "RHODE", "ISLAND", "OCEAN",
            "PENNSYLVANIA", "TEXAS", "FLORIDA", "CALIFORNIA", "VERMONT", "MAINE", "NH"
        )

        internal fun extractPlateToken(cleanedUppercase: String): String? {
            val tokens = cleanedUppercase.split(' ').filter { it.isNotEmpty() }
            tokens.firstOrNull { token ->
                token.length in 5..8 &&
                    token.any { it.isDigit() } &&
                    token.any { it.isLetter() }
            }?.let { return it }

            for (i in 0 until tokens.size - 1) {
                val a = tokens[i]
                val b = tokens[i + 1]
                if (a.length in 2..4 && b.length in 2..4) {
                    val combined = a + b
                    if (combined.length in 5..8 &&
                        combined.any { it.isDigit() } &&
                        combined.any { it.isLetter() }
                    ) {
                        return combined
                    }
                }
            }
            return null
        }

        /**
         * Extracts a 3–4 char partial plate fragment (ignoring state header words) when glare,
         * tinted covers, or distance prevent a full 5–8 char read (supports Phase 3 Tier 2 matching).
         */
        internal fun extractPartialPlateFragment(cleanedUppercase: String): String? {
            val tokens = cleanedUppercase.split(' ').filter { it.isNotEmpty() && it !in STATE_WORDS }
            return tokens.firstOrNull { tok ->
                tok.length in 3..4 && tok.any { it.isDigit() } && (tok.any { it.isLetter() } || tok.length == 4)
            }
        }

        internal fun plateFormatScore(token: String): Int {
            val letters = token.count { it.isLetter() }
            val digits  = token.count { it.isDigit() }
            if (letters == 1 && digits >= 6) return 0
            return 1
        }
    }

    internal data class VoteResult(
        val primaryConsensus: String?,
        val confirmedTokens: List<String>
    )

    internal class OcrVoteBuffer(
        private val windowSize: Int = VOTE_WINDOW,
        private val minVotes: Int = VOTE_MIN_COUNT
    ) {
        private val lock = Any()
        private val recentFrameTokens = ArrayDeque<Set<String>>()

        fun registerFrame(tokensInFrame: List<String>): VoteResult = synchronized(lock) {
            val uniqueInFrame = tokensInFrame.toSet()
            recentFrameTokens.addLast(uniqueInFrame)
            while (recentFrameTokens.size > windowSize) recentFrameTokens.removeFirst()

            val counts = mutableMapOf<String, Int>()
            for (frameSet in recentFrameTokens) {
                for (t in frameSet) {
                    counts[t] = (counts[t] ?: 0) + 1
                }
            }

            val confirmed = counts.entries
                .filter { it.value >= minVotes }
                .sortedWith(
                    compareByDescending<Map.Entry<String, Int>> { it.value }
                        .thenByDescending { if (it.key == tokensInFrame.firstOrNull()) 1 else 0 }
                        .thenByDescending { plateFormatScore(it.key) }
                )
                .map { it.key }

            VoteResult(
                primaryConsensus = confirmed.firstOrNull(),
                confirmedTokens = confirmed
            )
        }

        fun reset() = synchronized(lock) { recentFrameTokens.clear() }
    }
}
