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
| Voice control | Vosk Android (`vosk-model-small-en-us`, offline constrained grammar) + Android TTS |
| Database | Room (SQLite) v6, explicit versioned migrations (`MIGRATION_4_5`, `MIGRATION_5_6`) |
| Maps | OSMDroid (online tile download + local offline tile cache) |

## Building

1. Open Android Studio → **File → Open** → select this repo's `android/` folder.
2. Let Gradle sync.
3. Connect a device (USB debugging enabled) and hit **Run**.

Command line:
```bash
cd android
./gradlew assembleDebug
```

**Note on model files:** The large offline ML & speech binaries are gitignored and live locally in `android/app/src/main/assets/`:
- `yolov8_license_plate.tflite` — custom YOLOv8n plate detector (~12 MB)
- `mmc_classifier.tflite` — EfficientNet-B0 196-class vehicle fallback (~17 MB)
- `model-en-us/` — unpacked `vosk-model-small-en-us-0.15` directory (with `uuid` file) for offline `"Jolt..."` wake-phrase & command recognition

The app degrades gracefully if optional models (`mmc_classifier.tflite` or `model-en-us/`) are absent on a fresh clone.

## Hands-Free Voice Commands & Tactical HUD

Say **"Jolt"** (or **"Hey Jolt"**) while on the **Main** tab:
- **One-shot flagging:** `"Jolt, flag weaving"`, `"Jolt, flag white SUV cutting and asshole"`, `"Jolt, flag tailgating"`
- **Two-step conversational flagging:** `"Jolt, flag that car"` → Jolt reads the plate aloud (`"Plate L T 2 3 7 9. Reason?"`) → reply within 7 seconds (`"Line cutter"`, `"Unsteady throttle"`, `"No plate"`, etc.)
- **Positive Angel mark:** `"Jolt, angel that car"` (permanent green goodwill badge alongside any history)
- **Corrections & mute:** `"Jolt, wrong car"` / `"Jolt, cancel"` (undoes the last flag or suppresses an active alert), `"Jolt, mute alerts"` / `"Jolt, unmute alerts"`
- **Repeat-encounter Tactical HUD:** Displays severity color (**Maroon / Red / Orange / Yellow / Angel Green**), behavior code, incident count pips (`● ●`), Angel halo (`★ ANGEL`), match certainty, and concise spoken TTS alerts (*"Red. Weaver."*).

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
