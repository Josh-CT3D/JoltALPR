# Jolt ALPR

A vehicle-mounted Automatic License Plate Reader (ALPR) for US plates, running as a native
Android app entirely on-device. No internet connection required for any core function.

**Use case:** personal road-neighbor awareness. Flag a vehicle by hands-free voice command (or
touch fallback) for risky or discourteous behavior (`Weaving`, `Unsteady Throttle`, `Cutting`,
`Tailgating`, `Speed`, `No Lane`, `No Plate`, `Asshole`) or mark courteous drivers as **`★ Angel`**.
When a previously recorded plate is recognized again, Jolt raises a severity-coded Tactical HUD
(**Maroon / Red / Orange / Yellow / Angel Green**), haptic pattern, and spoken TTS alert (*"Red.
Weaver."*) — entirely from a local on-device Room database.

## How it works

The app watches the road through the phone's 5× telephoto lens, runs a locally-trained YOLOv8n
model on the GPU delegate to detect up to 3 license plates per frame, and reads them with
on-device OCR (Google ML Kit, fully offline) using multi-frame consensus voting and confusion-folded
plate matching. When a plate can't be read — glare, angle, distance, or tinted cover — a second
GPU-accelerated model (EfficientNet-B0) classifies the vehicle's make and model from a
plate-anchored crop. Hands-free voice commands are spotted locally at 16 kHz via Vosk constrained
grammar with zero cloud calls.

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
- **Standalone wake + 8s command window:** Say `"Jolt"` → chime + yellow `🎙 LISTENING — SAY COMMAND…` badge → say `"weaving"`, `"cutting"`, `"angel"`, `"cancel"`, etc.
- **One-shot flagging:** `"Jolt, flag weaving"`, `"Jolt, flag white Honda SUV cutting and asshole"`, `"Jolt, flag tailgating"`
- **Two-step conversational flagging:** `"Jolt, flag that car"` → Jolt reads the plate aloud (`"Plate L T 2 3 7 9. Reason?"`) → reply within 7 seconds (`"Line cutter"`, `"Unsteady throttle"`, `"No plate"`, etc.)
- **Positive Angel mark:** `"Jolt, angel that car"` (permanent green goodwill badge alongside any history)
- **Corrections & mute:** `"Jolt, wrong car"` / `"Jolt, cancel"` (undoes the last flag or suppresses an active alert), `"Jolt, mute alerts"` / `"Jolt, alerts on"`
- **Repeat-encounter Tactical HUD:** Displays severity color (**Maroon / Red / Orange / Yellow / Angel Green**), behavior code, incident count pips (`● ●`), Angel halo (`★ ANGEL`), match certainty, and concise spoken TTS alerts (*"Red. Weaver."*).

## Field Telemetry & Soak Logger

For untethered road validation without USB `adb`, toggle **Field Telemetry Logger** ON in the **Export** tab before driving. It records per-frame YOLO/OCR/MMC latencies, GPU delegate status, dropped frames, `PowerManager` thermal status & 10s thermal headroom, battery % and °C, Java/native heap MB, GPS speed (mph), and voice/flag actions to disk, and exports a self-contained `.txt` report with an Executive Summary Header to `Downloads/`.

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

Core plate detection, multi-plate OCR voting, confusion-tolerant matching, Tactical HUD & TTS
alerts, offline Vosk hands-free voice control, 8-tag behavior & Angel reputation system (Room v6),
OSMDroid street-map pin caching, DB+crops ZIP backup, and the Field Telemetry Logger are
implemented, unit-tested, and verified on-device (Pixel 10). Real-world road soak & telemetry
validation is in progress.
