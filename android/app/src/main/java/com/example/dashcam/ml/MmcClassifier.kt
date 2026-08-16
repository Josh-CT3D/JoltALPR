package com.ct3d.jolt.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.exp

/**
 * Make/Model/Colour classifier — Phase 5 fallback for when the plate can't be read.
 *
 * Model: EfficientNet-B0 fine-tuned on Stanford Cars (196 classes), trained locally on the
 * RTX 4060 Ti to 88.1% val top-1 and exported via litert-torch.
 *
 * Input:  [1, 224, 224, 3] float32, NHWC — deliberately channel-LAST so the pixel loop here
 *         mirrors [YoloV8Detector]'s. Values must be ImageNet-normalised, NOT plain 0–1:
 *         the training transform was Resize(224,224) → ToTensor() → Normalize(mean, std).
 *         Getting this wrong doesn't crash, it just returns confident nonsense.
 * Output: [1, 196] raw logits (no softmax in-graph) → softmax here for a usable confidence.
 *
 * THREADING (R1): like [YoloV8Detector], the TFLite GPU delegate binds its EGL context to the
 * thread that builds the Interpreter, so initialize() AND every classify() call must happen on
 * TelephotoAnalyzer's single `jolt-pipeline` thread. Never call this from the ML Kit OCR callback
 * (that runs on the main thread) — dispatch back onto the pipeline dispatcher first.
 */
class MmcClassifier(private val context: Context) {

    data class Prediction(val label: String, val confidence: Float)

    private var interpreter: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var labels: List<String> = emptyList()

    /** True once initialize() has run and the model+labels actually loaded. */
    @Volatile
    var isAvailable: Boolean = false
        private set

    private val inputSize = 224
    private val pixelSize = 3

    // ImageNet normalisation — must match train_mmc.py's transforms.Normalize exactly.
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std  = floatArrayOf(0.229f, 0.224f, 0.225f)

    // Reusable scratch, allocated once (A3 pattern) — avoids ~1MB of per-frame churn.
    // Guarded by @Synchronized on initialize/classify/close.
    private lateinit var inputBuffer: ByteBuffer
    private lateinit var intValues: IntArray
    private lateinit var outputBuffer: Array<FloatArray>
    private lateinit var resizedBitmap: Bitmap
    private lateinit var resizeCanvas: Canvas
    private val resizeMatrix = Matrix()
    private val resizePaint = Paint(Paint.FILTER_BITMAP_FLAG)

    /**
     * Load model + labels. Safe to call when the assets are absent: `mmc_classifier.tflite` is
     * gitignored ("large binaries, distribute separately"), so a fresh clone legitimately won't
     * have it. In that case we log and leave [isAvailable] false rather than throwing — the app
     * then behaves exactly as it did pre-Phase-5 instead of crashing on launch.
     */
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
            Log.i(TAG, "MMC input ${inShape.contentToString()} output ${outShape.contentToString()}, " +
                       "${labels.size} labels")

            // Fail loudly on a model/label mismatch rather than silently mislabelling every
            // prediction (e.g. re-exported with a different class count).
            val numClasses = outShape.last()
            if (numClasses != labels.size) {
                Log.e(TAG, "Model outputs $numClasses classes but $LABELS_ASSET has ${labels.size} " +
                           "— refusing to load (labels would be wrong).")
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
            // Most likely cause: mmc_classifier.tflite not present in assets.
            Log.w(TAG, "MMC unavailable (${e.javaClass.simpleName}: ${e.message}) — " +
                       "plate OCR still works, MMC fallback disabled")
            isAvailable = false
        }
    }

    /** GPU delegate first, CPU fallback — same wrapping rationale as YoloV8Detector.A1. */
    private fun createInterpreter(modelBuffer: MappedByteBuffer): Interpreter {
        try {
            val delegate = GpuDelegate(GpuDelegate.Options().apply { setPrecisionLossAllowed(true) })
            gpuDelegate = delegate
            val gpuInterpreter = Interpreter(modelBuffer, Interpreter.Options().apply { addDelegate(delegate) })
            Log.i(TAG, "MMC GPU delegate enabled")
            return gpuInterpreter
        } catch (e: Exception) {
            Log.w(TAG, "MMC GPU delegate unavailable, CPU fallback: ${e.message}")
            gpuDelegate?.close()
            gpuDelegate = null
        }
        val cpuInterpreter = Interpreter(modelBuffer, Interpreter.Options().apply { setNumThreads(4) })
        Log.i(TAG, "MMC model loaded — CPU, 4 threads")
        return cpuInterpreter
    }

    /**
     * Classify a vehicle image. Returns null when the model isn't loaded, inference fails, or the
     * top-1 softmax confidence is below [minConfidence] — callers treat null as "no MMC to show".
     *
     * MUST run on the `jolt-pipeline` thread (see class kdoc).
     */
    @Synchronized
    fun classify(bitmap: Bitmap, minConfidence: Float = MIN_MMC_CONFIDENCE): Prediction? {
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

            // Log top-3 so field misclassifications are diagnosable from a logcat capture.
            val top3 = probs.indices.sortedByDescending { probs[it] }.take(3)
            Log.d(TAG, "MMC top3: " + top3.joinToString(", ") {
                "${displayLabel(labels[it])}=${"%.3f".format(probs[it])}"
            })

            val bestIdx = top3.first()
            val confidence = probs[bestIdx]
            if (confidence < minConfidence) {
                Log.d(TAG, "MMC below floor ($confidence < $minConfidence) — no fallback shown")
                null
            } else {
                Prediction(displayLabel(labels[bestIdx]), confidence)
            }
        } catch (e: Exception) {
            Log.e(TAG, "MMC inference failed: ${e.message}", e)
            null
        }
    }

    /**
     * Scale into the reused 224×224 bitmap, then fill the input buffer NHWC with ImageNet-
     * normalised floats. Mirrors train_mmc.py's val transform: a plain Resize to a square
     * (aspect ratio is intentionally NOT preserved — that's how the model was trained).
     */
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

    /** Numerically-stable softmax (subtract max before exp). */
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

    /** "Chevrolet_Express_Van_2007" -> "Chevrolet Express Van 2007" for on-screen display. */
    private fun displayLabel(raw: String): String = raw.replace('_', ' ')

    private fun loadModelFile(filename: String): MappedByteBuffer {
        val fd = context.assets.openFd(filename)
        FileInputStream(fd.fileDescriptor).use { input ->
            return input.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
    }

    @Synchronized
    fun close() {
        interpreter?.close()
        interpreter = null
        gpuDelegate?.close()
        gpuDelegate = null
        if (::resizedBitmap.isInitialized && !resizedBitmap.isRecycled) resizedBitmap.recycle()
        isAvailable = false
        Log.i(TAG, "MmcClassifier closed")
    }

    companion object {
        private const val TAG = "MmcClassifier"
        private const val MODEL_ASSET  = "mmc_classifier.tflite"
        private const val LABELS_ASSET = "mmc_class_names.txt"

        /**
         * Top-1 softmax floor for showing an MMC guess. 196 fine-grained classes means the model
         * legitimately spreads probability across similar trims, so this is deliberately low —
         * a wrong "Toyota Camry 2011" is cheap, a missing fallback is the thing we're avoiding.
         * Tune from the "MMC top3" logs after a real drive.
         */
        const val MIN_MMC_CONFIDENCE = 0.35f
    }
}
