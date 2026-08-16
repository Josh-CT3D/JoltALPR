# Jolt ALPR

A vehicle-mounted Automatic License Plate Reader (ALPR) for US plates, running as a native
Android app entirely on-device. No internet connection required for any core function.

**Use case:** personal safety. Flag a driver who cuts you off or drives dangerously; if their
plate is ever read again, Jolt raises an on-screen alert — a pulsing yellow border, a warning
card, and a haptic buzz — entirely from a local on-device database. No cloud, no accounts, no
network calls.

## How it works

The app watches the road through the phone's telephoto lens, runs a locally-trained YOLOv8
model to find license plates in each frame, and reads them with on-device OCR (Google ML Kit,
fully offline). When a plate can't be read — glare, angle, distance — a second model falls back
to guessing the vehicle's make and model instead of showing nothing. A **FLAG BAD DRIVER**
button saves the current plate/vehicle guess, a GPS fix, and a photo of the plate to a local
database. Every subsequent plate read is checked against that database in real time.

## Hardware

| | |
|---|---|
| Primary test device | Google Pixel 10 (Tensor G5) |
| Minimum supported | Pixel 6+ (Tensor G1), Android 12 / API 31 |
| Mount | Dashboard or windshield, camera facing forward |
| Capture direction | Same-direction traffic only (rear plates) |
| Conditions | Daytime only — no IR illuminator, night not currently supported |

## Tech stack

| Layer | Technology |
|---|---|
| Language | Kotlin, MVVM + StateFlow |
| UI | Jetpack Compose (Material3), portrait, dark theme |
| Camera | CameraX, 5× telephoto zoom on supported lenses |
| ML inference | LiteRT (TFLite) with GPU delegate, CPU fallback |
| Plate detection | Custom YOLOv8n (see Models below) |
| OCR | Google ML Kit Text Recognition V2 (bundled, offline) |
| Vehicle fallback | Custom EfficientNet-B0 (see Models below) |
| Database | Room (SQLite), explicit versioned migrations |
| Maps | OSMDroid, offline tiles |

## Building

1. Open Android Studio → **File → Open** → select this repo's `android/` folder.
2. Let Gradle sync.
3. Connect a device (USB debugging enabled) and hit **Run**.

Command line:
```bash
cd android
./gradlew assembleDebug
```

**Note on model files:** `mmc_classifier.tflite` (the vehicle-fallback model, ~17 MB) is
gitignored and not in this repo — it needs to be copied into
`android/app/src/main/assets/` separately. The app degrades gracefully without it: plate
detection and OCR work normally, the vehicle-fallback feature is simply disabled.
`yolov8_license_plate.tflite` (the plate detector) is also gitignored for the same reason
(large binary, distributed separately).

## Models

**License plate detector** — custom YOLOv8n, trained locally (not from a pretrained/cloud
service — an earlier attempt to source a plate-detection model from Roboflow never produced a
working result). 81 epochs on 7,057 training images from a Kaggle dataset, mAP50 = 0.972.

**Vehicle make/model fallback** — custom EfficientNet-B0, fine-tuned on the Stanford Cars
dataset (196 classes), 88.1% validation top-1 accuracy. Fires only when the plate detector finds
no plate, or OCR fails to extract a usable token, so a flagged vehicle is never a total loss even
when the plate itself is unreadable.

Both models train on a local GPU via WSL2 — no cloud training service is used anywhere in this
project.

## Status

Core plate detection, OCR, known-bad-driver alerting, GPS mapping, and the vehicle-fallback
model are working and verified on-device. Training-data collection mode and its Google Drive
export path are implemented but not yet field-validated. See in-repo commit history for the
detailed changelog.
