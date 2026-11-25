package com.ct3d.jolt.camera

import android.content.Context
import android.graphics.*
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
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

/**
 * Immutable plate box in fractional (0..1) frame coordinates, drawn by the UI overlay.
 *
 * Replaces mutable android.graphics.RectF in UI/pipeline state: RectF is mutable, so Compose
 * treats state containing it as unstable and recomposes the Main screen every frame. A data class
 * of vals (all primitives) is inferred stable, so the overlay only recomposes when boxes change.
 */
data class PlateBox(val left: Float, val top: Float, val right: Float, val bottom: Float)

/**
 * CameraX ImageAnalysis.Analyzer running at 2 FPS (see ANALYSIS_INTERVAL_MS).
 *
 * Pipeline (Phase 2):
 *  1. YOLOv8 license-plate detection on the full frame (A6: ROI crop removed — see below)
 *  2a. Plate found → ML Kit Text Recognition V2 → validate US plate regex → emit OCR string
 *  2b. No plate    → clear display (Option A: chip disappears)
 *
 * Threading (R1): the entire pipeline — YOLO detector init AND every detect() call — runs on a
 * single dedicated thread (pipelineExecutor). The TFLite GPU delegate binds its EGL context to the
 * thread that created the interpreter, so running detect() on any other thread throws
 * "GpuDelegate must run on the same thread where it was initialized" and detection silently
 * returns nothing. Confining the pipeline to one thread guarantees init and inference share it.
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
        // All voted plate tokens confirmed in the current frame (supports multi-plate scenes
        // so a known-bad plate in an adjacent lane still triggers alerts).
        val confirmedPlateTokens: List<String> = emptyList(),
        val fpsTimestamp: Long = 0L,
        val processingTimeMs: Long = 0,
        val lastStatusMessage: String = "Pipeline Idle",
        // Plate boxes as fractions of the full camera frame (0.0–1.0).
        // The UI draws green overlay rectangles from these. Immutable PlateBox (not RectF) keeps
        // this state Compose-stable so the Main screen doesn't recompose every frame.
        val plateBoxes: List<PlateBox> = emptyList(),
        // Analyzed frame dimensions for accurate FILL_CENTER bounding-box projection in the UI.
        val frameWidth: Int = 0,
        val frameHeight: Int = 0,
        // Most recent successful plate crop bitmap (upscaled). Null when no plate in view.
        val lastPlateCrop: Bitmap? = null
    )

    private val _pipelineState = MutableStateFlow(PipelineState())
    val pipelineState: StateFlow<PipelineState> = _pipelineState

    private var lastAnalyzedTimestamp = 0L

    // R1: single dedicated pipeline thread. The GPU delegate's EGL context is bound to the thread
    // that created the interpreter, so the lazy yoloDetector init AND every detect() call must run
    // on the same thread. Confining every frame coroutine here guarantees that.
    private val pipelineExecutor = Executors.newSingleThreadExecutor { r -> Thread(r, "jolt-pipeline") }
    private val frameDispatcher = pipelineExecutor.asCoroutineDispatcher()

    // R2: the timestamp gate limits launch rate, not concurrency. With a serial pipeline (R1) a
    // frame that takes longer than ANALYSIS_INTERVAL_MS would otherwise queue unboundedly. Drop
    // any frame that arrives while the pipeline is still draining the previous one.
    private val pipelineBusy = AtomicBoolean(false)
    private val busyDropsSinceLastFrame = AtomicInteger(0)
    private var previousExecutedFrameAtMs = 0L

    // A15: multi-frame OCR voting. A single ML Kit misread (LT2379 -> LT2879) must not immediately
    // become the active plate and fire a false known-bad-driver alert.
    private val voteBuffer = OcrVoteBuffer(VOTE_WINDOW, VOTE_MIN_COUNT)

    // Phase 6: receives (fullFrameBitmap, allConfidentBoxes) for training data collection
    var trainingCallback: ((Bitmap, List<RectF>) -> Unit)? = null

    // Field validation telemetry callback — invoked on jolt-pipeline after each analyzed frame.
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

    // Phase 5 MMC fallback. Lazy like yoloDetector, and for the same R1 reason: first touch happens
    // inside a pipeline coroutine, so the interpreter is created on the jolt-pipeline thread that
    // every classify() call also runs on. initialize() never throws — if the (gitignored) model
    // asset is absent, isAvailable stays false and the pipeline behaves exactly as it did pre-P5.
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

        // Analysis-rate gate — release frame immediately if too soon (see ANALYSIS_INTERVAL_MS)
        if (currentMillis - lastAnalyzedTimestamp < ANALYSIS_INTERVAL_MS) {
            image.close()
            return
        }
        lastAnalyzedTimestamp = currentMillis

        // R2: drop this frame if the previous one is still being processed (serial pipeline).
        // Reset in the coroutine's finally below.
        if (!pipelineBusy.compareAndSet(false, true)) {
            busyDropsSinceLastFrame.incrementAndGet()
            image.close()
            return
        }

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
            pipelineBusy.set(false) // never launched the coroutine — release the gate here
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

    /**
     * Phase 2 pipeline — single-class license plate detection.
     * 1. Run YOLOv8 on the full frame (A6: no ROI crop — see the note in this method's body).
     * 2a. Plate found → compute fractional overlay boxes → crop → ML Kit OCR.
     * 2b. No plate    → clear display (Option A).
     */
    private suspend fun executeHybridMlPipeline(frameBitmap: Bitmap): PipelineMetrics {
        val roiBitmap = frameBitmap
        val validPlates: List<Detection>
        val trainingBoxes: List<RectF>
        // Per-plate OCR inputs prepared synchronously while roiBitmap is alive.
        data class PlateCandidate(val det: Detection, val ocrCrop: Bitmap, val persistCrop: Bitmap)
        val candidates = mutableListOf<PlateCandidate>()
        var mmcFallback: Bitmap? = null
        var yoloElapsedMs = 0L
        var mmcElapsedMs = 0L
        var mmcTrigger = "NONE"
        var mmcDetail: MmcClassifier.DetailedPrediction? = null
        var boxCount = 0
        var topConf = 0f

        try {
            _pipelineState.value = _pipelineState.value.copy(
                lastStatusMessage = "Scanning...",
                frameWidth = roiBitmap.width,
                frameHeight = roiBitmap.height
            )
            val yoloStart = System.currentTimeMillis()
            val detections = runYoloDetection(roiBitmap)
            yoloElapsedMs = System.currentTimeMillis() - yoloStart
            Log.d(TAG, "YOLO: ${detections.size} plate detections (threshold 0.25)")

            validPlates = detections
                .filter { it.confidence > 0.25f }
                .sortedByDescending { it.confidence }
                .take(MAX_OCR_PLATES_PER_FRAME)
            boxCount = validPlates.size
            topConf = validPlates.firstOrNull()?.confidence ?: 0f

            if (validPlates.isNotEmpty()) {
                val fractionalBoxes = computeFractionalBoxes(validPlates, roiBitmap)
                _pipelineState.value = _pipelineState.value.copy(
                    plateBoxes = fractionalBoxes,
                    lastStatusMessage = "Plate found — running OCR..."
                )

                val bestPlate = validPlates.first()
                Log.d(TAG, "Best plate box: left=${bestPlate.box.left} top=${bestPlate.box.top} " +
                           "right=${bestPlate.box.right} bottom=${bestPlate.box.bottom} " +
                           "conf=${bestPlate.confidence}  ROI=${roiBitmap.width}×${roiBitmap.height}")

                // A19: collect high-confidence frames with ALL confident boxes in frame so YOLO
                // training labels don't omit secondary visible plates.
                trainingBoxes = detections
                    .filter { it.confidence >= TRAINING_MIN_CONFIDENCE }
                    .map { it.box }
                if (trainingBoxes.isNotEmpty()) {
                    trainingCallback?.invoke(roiBitmap, trainingBoxes)
                }

                for (det in validPlates) {
                    val rawCrop = cropBitmapToDetection(roiBitmap, det.box) ?: continue
                    val ocrCrop = upscaleCrop(rawCrop)
                    val persistCopy = upscaleCrop(rawCrop)
                    rawCrop.recycle()
                    candidates.add(PlateCandidate(det, ocrCrop, persistCopy))
                }

                if (candidates.isNotEmpty()) {
                    if (mmcClassifier.isAvailable) {
                        mmcFallback = buildMmcInput(roiBitmap, bestPlate.box)
                    }
                } else {
                    Log.w(TAG, "All plate crops null — clearing display.")
                    clearDetectionState()
                }
            } else {
                // Phase 5 trigger (a): no plate box at all — identify the vehicle instead of
                // showing nothing. We're already on the pipeline thread, so classify inline.
                Log.d(TAG, "No plate detected — trying MMC fallback (Option A).")
                resetVoteBuffer() // plate left frame — start voting fresh for the next one
                clearDetectionState()
                if (mmcClassifier.isAvailable) {
                    mmcTrigger = "NO_PLATE_CENTER"
                    val mmcStart = System.currentTimeMillis()
                    mmcDetail = classifyAndPublishMmc(buildMmcInput(roiBitmap))
                    mmcElapsedMs = System.currentTimeMillis() - mmcStart
                }
            }
        } finally {
            // A4(b): all crops/snapshots are already taken synchronously — recycle full frame now.
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

        // Run ML Kit OCR sequentially across the cropped candidates, resuming back onto
        // frameDispatcher (jolt-pipeline). Running sequentially before returning keeps R1/R2/C1
        // ordered by construction (no stale frame N callback overwriting frame N+1).
        val extracted = mutableListOf<Pair<PlateCandidate, String>>()
        val ocrStart = System.currentTimeMillis()
        try {
            for (candidate in candidates) {
                val raw = processOcr(candidate.ocrCrop)
                val clean = raw?.uppercase()?.filter { it.isLetterOrDigit() || it == ' ' } ?: ""
                val token = if (clean.isNotEmpty()) extractPlateToken(clean) else null
                if (token != null) {
                    extracted.add(candidate to token)
                }
            }
        } finally {
            // Ensure every OCR crop is recycled even on cancellation/exception.
            for (c in candidates) recycleQuietly(c.ocrCrop)
        }
        val ocrElapsedMs = System.currentTimeMillis() - ocrStart

        if (extracted.isNotEmpty()) {
            recycleQuietly(mmcFallback)
            val chosenPersist = extracted.first().first.persistCrop
            for (c in candidates) {
                if (c.persistCrop !== chosenPersist) recycleQuietly(c.persistCrop)
            }
            // R4: hand ownership of chosenPersist to the ViewModel (never recycle here).
            _pipelineState.value = _pipelineState.value.copy(lastPlateCrop = chosenPersist)

            val tokensInFrame = extracted.map { it.second }
            val voteResult = registerAndVoteFrame(tokensInFrame)
            val consensus = voteResult.primaryConsensus
            if (consensus != null) {
                Log.i(TAG, "OCR consensus: '$consensus' (frame tokens=$tokensInFrame, confirmed=${voteResult.confirmedTokens})")
                _pipelineState.value = _pipelineState.value.copy(
                    activePlateocr       = consensus,
                    confirmedPlateTokens = voteResult.confirmedTokens,
                    activeVehicleMmc     = null,
                    lastStatusMessage    = "OCR: $consensus"
                )
            } else {
                val pending = tokensInFrame.first()
                Log.d(TAG, "OCR pending consensus: '$pending' (frame tokens=$tokensInFrame)")
                _pipelineState.value = _pipelineState.value.copy(
                    activeVehicleMmc     = null, // C5: clear stale MMC while confirming plate
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
            // Path (b): plate box(es) present, OCR yielded no usable plate token.
            // Publish the primary box's crop (photo of the unreadable/obscured plate) and run MMC.
            val fallbackPersist = candidates.first().persistCrop
            for (c in candidates) {
                if (c.persistCrop !== fallbackPersist) recycleQuietly(c.persistCrop)
            }
            _pipelineState.value = _pipelineState.value.copy(
                activePlateocr       = null,
                confirmedPlateTokens = emptyList(),
                lastPlateCrop        = fallbackPersist,
                lastStatusMessage    = "Plate unreadable — trying MMC fallback..."
            )
            if (mmcFallback != null) {
                mmcTrigger = "OCR_FAIL_ANCHORED"
                val mmcStart = System.currentTimeMillis()
                mmcDetail = classifyAndPublishMmc(mmcFallback)
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

    /**
     * Suspends on ML Kit Text Recognition for [plateBitmap] and resumes on the caller's
     * dispatcher (`jolt-pipeline`), recycling [plateBitmap] on completion.
     */
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
            activePlateocr       = null,
            activeVehicleMmc     = null,
            confirmedPlateTokens = emptyList(),
            plateBoxes           = emptyList(),
            lastPlateCrop        = null, // R4: do NOT recycle here; ViewModel owns published crops
            lastStatusMessage    = "Scanning..."
        )
    }

    private fun registerAndVoteFrame(tokensInFrame: List<String>): VoteResult =
        voteBuffer.registerFrame(tokensInFrame)

    private fun resetVoteBuffer() = voteBuffer.reset()

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

    // ── Phase 5: MMC fallback ──────────────────────────────────────────────────────────────────

    /**
     * O9 / F2 — build the MMC input by cropping to roughly where the *vehicle* is, then scaling to
     * the model's 224² input.
     *
     * Why crop at all: the classifier was fine-tuned on Stanford Cars, i.e. tightly-framed single
     * vehicles. Feeding it a whole 480×640 scene put top-1 confidence at 0.07–0.14 — it essentially
     * never fired. Pointing the camera at an actual training image scored 0.32–0.49 with the
     * correct class, which isolated the cause as framing, not preprocessing.
     *
     * [plateBox] (frame pixel coords) is used when we have one: a US plate is ~1/6–1/8 of car
     * width, so scaling outward from the plate reconstructs Stanford-Cars-like framing from
     * geometry we already have — no extra model, no measurable latency. With no plate box we fall
     * back to a centre square, which at 5× telephoto usually holds the followed vehicle.
     *
     * Also keeps the async path cheap: the result is a ~200 KB bitmap, so the OCR callback can
     * still classify after [executeHybridMlPipeline]'s finally has recycled the full frame.
     */
    private fun buildMmcInput(src: Bitmap, plateBox: RectF? = null): Bitmap {
        val fw = src.width.toFloat()
        val fh = src.height.toFloat()

        val crop = if (plateBox != null && plateBox.width() > 1f && plateBox.height() > 1f) {
            val pw = plateBox.width()
            val ph = plateBox.height()
            val cx = plateBox.centerX()
            // Plate sits low-centre on a car: extend far more upward than downward.
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

        // Clamp to the frame, then guard against a degenerate rect (plate at the very edge).
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
        // Squash to a square, matching train_mmc.py's Resize((224,224)) — aspect ratio is
        // intentionally not preserved, because that's how the model was trained.
        val scaled = Bitmap.createScaledBitmap(source, MMC_INPUT_SIZE, MMC_INPUT_SIZE, true)
        // createBitmap may hand back the source instance; only recycle a genuinely new intermediate.
        if (cropped != null && cropped !== scaled && cropped !== src) cropped.recycle()
        return scaled
    }

    /**
     * Classify [vehicleBitmap] and publish the result as the MMC fallback. Takes ownership of the
     * bitmap and always recycles it. MUST run on the `jolt-pipeline` thread.
     *
     * Leaves state untouched when the classifier declines (below confidence floor / unavailable) —
     * callers have already cleared the display, so "no guess" simply shows nothing.
     */
    private fun classifyAndPublishMmc(vehicleBitmap: Bitmap): MmcClassifier.DetailedPrediction? {
        return try {
            if (_pipelineState.value.activePlateocr != null) {
                Log.d(TAG, "MMC fallback dropped — a plate consensus is already displayed")
                return null
            }
            val detail = mmcClassifier.classifyDetailed(vehicleBitmap) ?: return null
            val prediction = detail.published
            if (prediction != null) {
                Log.i(TAG, "MMC fallback: ${prediction.label} (conf=${"%.3f".format(prediction.confidence)})")
                _pipelineState.value = _pipelineState.value.copy(
                    activePlateocr    = null,   // mutually exclusive with MMC (DriverLog stores one or the other)
                    activeVehicleMmc  = prediction.label,
                    lastStatusMessage = "MMC: ${prediction.label}"
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
            mmcClassifier.close()
            Log.i(TAG, "TelephotoAnalyzer cleaned up successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error during cleanup: ${e.message}", e)
        }
        // R1: shut down the dedicated pipeline thread (also stops its executor).
        try {
            frameDispatcher.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error closing pipeline dispatcher: ${e.message}", e)
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

        // Phase 5: MMC model input edge. Snapshots are pre-scaled to this so the async OCR-failure
        // path holds ~200 KB instead of a full frame (MmcClassifier rescales defensively anyway).
        private const val MMC_INPUT_SIZE = 224

        // O9 / F2: plate-box-anchored crop geometry. A US plate is ~1/6–1/8 of car width and sits
        // low-centre, so growing outward from the box approximates a Stanford-Cars framing.
        // 6×plate-width by 7×plate-height lands near 1.7:1 (plates are ~2:1), i.e. a landscape
        // car-photo shape. Tune from the "MMC top3" logs after a real drive.
        private const val MMC_CROP_WIDTH_MULT = 6.0f
        private const val MMC_CROP_UP_MULT    = 4.5f
        private const val MMC_CROP_DOWN_MULT  = 1.5f

        private const val MAX_OCR_PLATES_PER_FRAME = 3

        /**
         * Extract a US-style plate token from cleaned uppercase OCR text.
         *
         * 1. Prefer a single space-delimited token of 5–8 chars containing >=1 digit and >=1 letter
         *    (e.g. "CONNECTICUT LT2379 CONSTITUTION STATE" -> "LT2379").
         * 2. Fallback for space/hyphen-split plates (e.g. "ABC 1234" or "123 ABC"): join adjacent
         *    short tokens (each 2–4 chars) whose combined length is 5–8 with >=1 digit and >=1 letter.
         */
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
         * A17: loose US-plate plausibility, used only as a voting tiebreaker (never a hard filter).
         * Returns 1 for plausible, 0 for digit-heavy OCR artifacts (e.g. "1234567X").
         */
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
