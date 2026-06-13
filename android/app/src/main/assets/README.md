# TFLite Model Assets

This directory contains the TensorFlow Lite models for offline license plate detection.

## Required Files

Place the following files in this directory:

### 1. `yolov8_license_plate.tflite`
- **Source:** Roboflow license-plate-recognition-rxg4e model
- **Format:** TensorFlow Lite (Float32 or Quantized INT8)
- **How to get:**
  1. Go to https://app.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e
  2. Click "Deploy" → "Export"
  3. Select "TensorFlow Lite" format
  4. Download and extract the ZIP
  5. Rename `model.tflite` to `yolov8_license_plate.tflite`
  6. Place in this directory

### 2. `labels.txt` (Optional)
- Contains class labels (e.g., "license_plate", "vehicle")
- Usually included in the Roboflow export ZIP
- Not currently used by the app but good for reference

## Model Specifications

**Expected Input:**
- Shape: [1, 640, 640, 3]
- Type: Float32
- Format: RGB normalized to [0, 1]

**Expected Output:**
- Shape: [1, 8400, 6]
- Format: [x_center, y_center, width, height, confidence, class_id]

**Classes:**
- 0: license_plate
- 1: vehicle (or car)

## Performance

The model runs with:
- **GPU/NPU acceleration** via LiteRT GPU delegate
- **Target latency:** ~100-150ms per frame on Pixel 10
- **Inference rate:** 1 FPS (thermal management)

## Troubleshooting

If the app crashes on model loading:
1. Verify `yolov8_license_plate.tflite` exists in this directory
2. Check Android Studio Logcat for "YoloV8Detector" errors
3. Ensure the model format is compatible (Float32 or Quantized INT8)
4. Try rebuilding the app (Clean + Rebuild)
