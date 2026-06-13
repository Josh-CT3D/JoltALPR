package com.ct3d.jolt.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

/**
 * YOLOv8 TFLite detector for license plate detection.
 * Runs on CPU (4 threads) — sufficient for 1fps on Pixel 10 Tensor G5.
 *
 * Model input:  [1, 640, 640, 3]  float32 RGB normalized to [0, 1]
 * Model output: [1, 5, 8400]      transposed format — rows are attributes, columns are anchors
 *               attributes: [cx, cy, w, h, score]  (single class: license_plate)
 *               coordinates are in model input space (0–640), not normalized
 */
class YoloV8Detector(private val context: Context) {

    data class Detection(
        val box: RectF,
        val label: String,
        val confidence: Float
    )

    private var interpreter: Interpreter? = null
    private val labels = listOf("license_plate")

    // Model input size (YOLOv8 typically uses 640x640)
    private val inputSize = 640
    private val pixelSize = 3 // RGB
    private val imageStd = 255.0f // Normalize to [0, 1]

    fun initialize() {
        try {
            val modelBuffer = loadModelFile("yolov8_license_plate.tflite")

            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }

            interpreter = Interpreter(modelBuffer, options)

            Log.i(TAG, "YOLOv8 model loaded — CPU, 4 threads")
            Log.i(TAG, "Output shape: ${interpreter!!.getOutputTensor(0).shape().contentToString()}")

        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize YOLOv8 detector: ${e.message}", e)
            throw e
        }
    }

    /**
     * Run object detection on the given bitmap.
     * Returns list of detections (vehicles and license plates).
     *
     * @param bitmap Input image (any size - will be resized to 640x640)
     * @return List of Detection objects with bounding boxes and confidence scores
     */
    fun detect(bitmap: Bitmap, confidenceThreshold: Float = 0.25f): List<Detection> {
        val tflite = interpreter ?: run {
            Log.w(TAG, "Interpreter not initialized")
            return emptyList()
        }

        try {
            val inputBuffer = preprocessImage(bitmap)

            // Output shape: [1, 5, 8400] — transposed YOLOv8 format
            // Dimension 1 = attribute index: 0=cx, 1=cy, 2=w, 3=h, 4=score
            // Dimension 2 = anchor index (8400 candidate boxes)
            val outputBuffer = Array(1) { Array(5) { FloatArray(8400) } }

            tflite.run(inputBuffer, outputBuffer)

            val detections = parseOutput(
                outputBuffer[0],
                bitmap.width,
                bitmap.height,
                confidenceThreshold
            )

            // Apply Non-Maximum Suppression to remove duplicate detections
            val filteredDetections = applyNMS(detections, iouThreshold = 0.5f)

            Log.d(TAG, "Detected ${filteredDetections.size} objects (after NMS)")
            return filteredDetections

        } catch (e: Exception) {
            Log.e(TAG, "Detection failed: ${e.message}", e)
            return emptyList()
        }
    }

    /**
     * Preprocess bitmap for YOLOv8 input:
     * 1. Resize to 640x640
     * 2. Convert to RGB
     * 3. Normalize to [0, 1]
     * 4. Convert to ByteBuffer
     */
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        // Resize image
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        // Allocate ByteBuffer for model input
        val inputBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * pixelSize)
        inputBuffer.order(ByteOrder.nativeOrder())

        // Extract pixel values and normalize
        val intValues = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(intValues, 0, inputSize, 0, 0, inputSize, inputSize)

        var pixel = 0
        for (y in 0 until inputSize) {
            for (x in 0 until inputSize) {
                val value = intValues[pixel++]

                // Extract RGB channels and normalize to [0, 1]
                inputBuffer.putFloat(((value shr 16) and 0xFF) / imageStd) // R
                inputBuffer.putFloat(((value shr 8) and 0xFF) / imageStd)  // G
                inputBuffer.putFloat((value and 0xFF) / imageStd)          // B
            }
        }

        return inputBuffer
    }

    /**
     * Parse YOLOv8 transposed output [5][8400] into Detection objects.
     * output[0][i] = cx, [1][i] = cy, [2][i] = w, [3][i] = h, [4][i] = score (single class).
     * Coordinates are in model input space (0–640); scale back to original image dimensions.
     */
    private fun parseOutput(
        output: Array<FloatArray>,  // shape [5][8400]
        originalWidth: Int,
        originalHeight: Int,
        confidenceThreshold: Float
    ): List<Detection> {
        val detections = mutableListOf<Detection>()
        val numAnchors = output[0].size

        val scaleX = originalWidth.toFloat() / inputSize
        val scaleY = originalHeight.toFloat() / inputSize

        // Check whether coordinates are normalized (0–1) or in pixel space (0–640).
        // Roboflow TFLite models export normalized coords; standard Ultralytics exports use pixel space.
        // We sample the first high-confidence detection to determine which format this model uses.
        var formatLogged = false

        for (i in 0 until numAnchors) {
            val score = output[4][i]
            if (score < confidenceThreshold) continue

            val cx = output[0][i]
            val cy = output[1][i]
            val w  = output[2][i]
            val h  = output[3][i]

            if (!formatLogged) {
                Log.i(TAG, "First detection raw values: cx=$cx cy=$cy w=$w h=$h score=$score " +
                           "(if cx/cy/w/h are all <2.0, model outputs normalized 0–1 coords)")
                formatLogged = true
            }

            // If coordinates look normalized (max value < 2.0), scale by image dimensions directly.
            // Otherwise assume 0–640 pixel space and use scaleX/scaleY.
            val isNormalized = cx < 2.0f && cy < 2.0f && w < 2.0f && h < 2.0f
            val left: Float
            val top: Float
            val right: Float
            val bottom: Float
            if (isNormalized) {
                left   = (cx - w / 2f) * originalWidth
                top    = (cy - h / 2f) * originalHeight
                right  = (cx + w / 2f) * originalWidth
                bottom = (cy + h / 2f) * originalHeight
            } else {
                left   = (cx - w / 2f) * scaleX
                top    = (cy - h / 2f) * scaleY
                right  = (cx + w / 2f) * scaleX
                bottom = (cy + h / 2f) * scaleY
            }

            val box = RectF(
                left.coerceIn(0f, originalWidth.toFloat()),
                top.coerceIn(0f, originalHeight.toFloat()),
                right.coerceIn(0f, originalWidth.toFloat()),
                bottom.coerceIn(0f, originalHeight.toFloat())
            )

            detections.add(Detection(box, labels[0], score))
        }

        return detections
    }

    /**
     * Apply Non-Maximum Suppression (NMS) to remove overlapping detections.
     * Keeps only the detection with highest confidence when boxes overlap significantly.
     */
    private fun applyNMS(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        if (detections.isEmpty()) return emptyList()

        // Sort by confidence (highest first)
        val sortedDetections = detections.sortedByDescending { it.confidence }.toMutableList()
        val keepDetections = mutableListOf<Detection>()

        while (sortedDetections.isNotEmpty()) {
            val best = sortedDetections.removeAt(0)
            keepDetections.add(best)

            // Remove overlapping detections of the same class
            sortedDetections.removeAll { detection ->
                detection.label == best.label &&
                calculateIoU(best.box, detection.box) > iouThreshold
            }
        }

        return keepDetections
    }

    /**
     * Calculate Intersection over Union (IoU) between two bounding boxes.
     */
    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val intersectLeft = max(box1.left, box2.left)
        val intersectTop = max(box1.top, box2.top)
        val intersectRight = min(box1.right, box2.right)
        val intersectBottom = min(box1.bottom, box2.bottom)

        if (intersectRight <= intersectLeft || intersectBottom <= intersectTop) {
            return 0f // No intersection
        }

        val intersectArea = (intersectRight - intersectLeft) * (intersectBottom - intersectTop)
        val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
        val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)
        val unionArea = box1Area + box2Area - intersectArea

        return intersectArea / unionArea
    }

    /**
     * Load TFLite model file from assets folder.
     */
    private fun loadModelFile(filename: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Release resources when done.
     * Call this when camera is stopped or app is destroyed.
     */
    fun close() {
        interpreter?.close()
        interpreter = null
        Log.i(TAG, "YOLOv8 detector closed")
    }

    companion object {
        private const val TAG = "YoloV8Detector"
    }
}
