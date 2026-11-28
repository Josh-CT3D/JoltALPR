package com.ct3d.jolt.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.ct3d.jolt.data.StructuredAppearance
import com.ct3d.jolt.data.VehicleAppearance
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

/**
 * Make/Body/Color Vehicle Appearance Classifier — Phase 5 + Phase 3 Upgraded.
 *
 * Model: EfficientNet-B0 fine-tuned on Stanford Cars (196 classes) with highway-distance blur and
 * scale-jitter augmentation on RTX 4060 Ti, fused with:
 *  1. Marginalized `(Make, BodyStyle)` probability summation across all 196 softmax outputs (so
 *     modern SUVs/Sedans/Pickups that split probability across trims still resolve cleanly without
 *     emitting stale 1991–2012 trim years).
 *  2. Zero-copy HSV dominant body-paint color estimation over the reused 224×224 pixel buffer.
 *
 * THREADING (R1): initialize() AND every classify() call MUST happen on `jolt-pipeline`.
 */
class MmcClassifier(private val context: Context) {

    data class Prediction(
        val label: String,
        val confidence: Float,
        val appearance: StructuredAppearance = StructuredAppearance()
    )

    data class DetailedPrediction(
        val published: Prediction?,
        val top1Label: String,
        val top1Confidence: Float,
        val top3Summary: String,
        val appearance: StructuredAppearance = StructuredAppearance()
    )

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var labels: List<String> = emptyList()

    @Volatile
    var isAvailable: Boolean = false
        private set

    @Volatile
    var isUsingGpuDelegate: Boolean = false
        private set

    private val inputSize = 224
    private val pixelSize = 3

    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std  = floatArrayOf(0.229f, 0.224f, 0.225f)

    private lateinit var inputBuffer: ByteBuffer
    private lateinit var intValues: IntArray
    private lateinit var outputBuffer: Array<FloatArray>
    private lateinit var resizedBitmap: Bitmap
    private lateinit var resizeCanvas: Canvas
    private val resizeMatrix = Matrix()
    private val resizePaint = Paint(Paint.FILTER_BITMAP_FLAG)

    @Synchronized
    fun initialize() {
        try {
            labels = context.assets.open(LABELS_ASSET).bufferedReader().useLines { seq ->
                seq.map { it.trim() }.filter { it.isNotEmpty() }.toList()
            }
            if (labels.isEmpty()) {
                Log.w(TAG, "$LABELS_ASSET empty — MMC disabled")
                return
            }

            val modelBuffer = loadModelFile(MODEL_ASSET)
            interpreter = createInterpreter(modelBuffer)

            val outShape = interpreter!!.getOutputTensor(0).shape()
            val inShape  = interpreter!!.getInputTensor(0).shape()
            Log.i(TAG, "MMC input ${inShape.contentToString()} output ${outShape.contentToString()}, ${labels.size} labels")

            val numClasses = outShape.last()
            if (numClasses != labels.size) {
                Log.e(TAG, "Model outputs $numClasses classes but $LABELS_ASSET has ${labels.size} — refusing to load.")
                interpreter?.close(); interpreter = null
                gpuDelegate?.close(); gpuDelegate = null
                return
            }

            inputBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * pixelSize)
                .order(ByteOrder.nativeOrder())
            intValues = IntArray(inputSize * inputSize)
            outputBuffer = Array(1) { FloatArray(numClasses) }
            resizedBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
            resizeCanvas = Canvas(resizedBitmap)

            isAvailable = true
            Log.i(TAG, "MmcClassifier initialized successfully")
        } catch (e: Exception) {
            Log.w(TAG, "MMC unavailable (${e.javaClass.simpleName}: ${e.message}) — plate OCR still works")
            isAvailable = false
        }
    }

    private fun createInterpreter(modelBuffer: MappedByteBuffer): Interpreter {
        try {
            val delegate = GpuDelegate(GpuDelegate.Options().apply { setPrecisionLossAllowed(true) })
            gpuDelegate = delegate
            val gpuInterpreter = Interpreter(modelBuffer, Interpreter.Options().apply { addDelegate(delegate) })
            isUsingGpuDelegate = true
            Log.i(TAG, "MMC GPU delegate enabled")
            return gpuInterpreter
        } catch (e: Exception) {
            Log.w(TAG, "MMC GPU delegate unavailable, CPU fallback: ${e.message}")
            gpuDelegate?.close()
            gpuDelegate = null
            isUsingGpuDelegate = false
        }
        val cpuInterpreter = Interpreter(modelBuffer, Interpreter.Options().apply { setNumThreads(4) })
        Log.i(TAG, "MMC model loaded — CPU, 4 threads")
        return cpuInterpreter
    }

    @Synchronized
    fun classify(bitmap: Bitmap, minConfidence: Float = MIN_MMC_CONFIDENCE): Prediction? =
        classifyDetailed(bitmap, minConfidence)?.published

    /**
     * Runs EfficientNet-B0 on `jolt-pipeline`, marginalizes `(Make, BodyStyle)` probabilities,
     * and extracts dominant HSV body color from the same `224×224` pixel buffer.
     */
    @Synchronized
    fun classifyDetailed(bitmap: Bitmap, minConfidence: Float = MIN_MMC_CONFIDENCE): DetailedPrediction? {
        val tflite = interpreter ?: return null
        if (bitmap.isRecycled) {
            Log.w(TAG, "classify() got a recycled bitmap — skipping")
            return null
        }

        return try {
            preprocess(bitmap)
            tflite.run(inputBuffer, outputBuffer)

            val logits = outputBuffer[0]
            val probs = softmax(logits)

            // Zero-copy HSV color extraction over the already-populated 224×224 intValues buffer
            val paintColor = VehicleAppearance.estimateBodyColorFromPixels(intValues, inputSize, inputSize)
            val appearance = VehicleAppearance.fuseMmcProbabilitiesAndColor(labels, probs, paintColor)

            val top3 = probs.indices.sortedByDescending { probs[it] }.take(3)
            val summary = top3.joinToString("; ") {
                "${displayLabel(labels[it])}=${"%.3f".format(probs[it])}"
            }
            Log.d(TAG, "MMC top3: $summary | structured=${appearance.formatSummary()} (mk=${"%.2f".format(appearance.makeConfidence)}, bd=${"%.2f".format(appearance.bodyConfidence)}, col=${"%.2f".format(appearance.colorConfidence)})")

            val bestIdx = top3.first()
            val bestRawLabel = displayLabel(labels[bestIdx])
            val top1Confidence = probs[bestIdx]

            // Publish clean modern summary ("White Honda SUV" or "Silver SUV") whenever either:
            //  (a) top-1 class >= minConfidence (0.25), OR
            //  (b) marginalized BodyStyle probability is strong (>= 0.42) with Make or Color!
            val effectiveConf = maxOf(top1Confidence, appearance.bodyConfidence * 0.85f)
            val allowMarginalFallback = minConfidence <= MIN_MMC_CONFIDENCE
            val hasModernSummary = allowMarginalFallback && appearance.bodyStyle != null &&
                (appearance.make != null || (appearance.color != null && appearance.bodyConfidence >= 0.42f))

            val published = if (top1Confidence >= minConfidence || (hasModernSummary && appearance.bodyConfidence >= 0.42f)) {
                val cleanLabel = appearance.formatSummary() ?: bestRawLabel
                Prediction(label = cleanLabel, confidence = effectiveConf, appearance = appearance)
            } else {
                null
            }

            DetailedPrediction(
                published = published,
                top1Label = bestRawLabel,
                top1Confidence = top1Confidence,
                top3Summary = summary,
                appearance = appearance
            )
        } catch (e: Exception) {
            Log.e(TAG, "MMC inference failed: ${e.message}", e)
            null
        }
    }

    /** Fast color-only estimation when OCR succeeded and full MMC TFLite isn't needed. */
    @Synchronized
    fun estimateColorOnly(bitmap: Bitmap): Pair<String?, Float> {
        if (bitmap.isRecycled) return null to 0f
        return try {
            val w = bitmap.width
            val h = bitmap.height
            val px = IntArray(w * h)
            bitmap.getPixels(px, 0, w, 0, 0, w, h)
            VehicleAppearance.estimateBodyColorFromPixels(px, w, h)
        } catch (_: Exception) {
            null to 0f
        }
    }

    private fun preprocess(bitmap: Bitmap) {
        resizeMatrix.reset()
        resizeMatrix.setScale(inputSize.toFloat() / bitmap.width, inputSize.toFloat() / bitmap.height)
        resizeCanvas.drawBitmap(bitmap, resizeMatrix, resizePaint)
        resizedBitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

        inputBuffer.rewind()
        var p = 0
        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val v = intValues[p++]
                val r = ((v shr 16) and 0xFF) / 255f
                val g = ((v shr 8) and 0xFF) / 255f
                val b = (v and 0xFF) / 255f
                inputBuffer.putFloat((r - mean[0]) / std[0])
                inputBuffer.putFloat((g - mean[1]) / std[1])
                inputBuffer.putFloat((b - mean[2]) / std[2])
            }
        }
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val max = logits.max()
        var sum = 0f
        val out = FloatArray(logits.size)
        for (i in logits.indices) {
            val e = exp((logits[i] - max).toDouble()).toFloat()
            out[i] = e
            sum += e
        }
        if (sum > 0f) for (i in out.indices) out[i] /= sum
        return out
    }

    private fun displayLabel(raw: String): String = raw.replace('_', ' ')

    private fun loadModelFile(assetName: String): MappedByteBuffer {
        return context.assets.openFd(assetName).use { fd ->
            FileInputStream(fd.fileDescriptor).use { fis ->
                fis.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }

    @Synchronized
    fun close() {
        isAvailable = false
        isUsingGpuDelegate = false
        interpreter?.close()
        interpreter = null
        gpuDelegate?.close()
        gpuDelegate = null
        Log.i(TAG, "MmcClassifier closed")
    }

    companion object {
        private const val TAG = "MmcClassifier"
        private const val MODEL_ASSET = "mmc_classifier.tflite"
        private const val LABELS_ASSET = "mmc_class_names.txt"
        const val MIN_MMC_CONFIDENCE = 0.25f
    }
}
