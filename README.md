# Jolt ALPR

A vehicle-mounted Automatic License Plate Reader (ALPR) and personal road-neighbor awareness system
for US plates, running as a native Android app entirely on-device.

**Use case:** personal road-neighbor memory ("*Awareness enables distance; distance creates safety*").
Flag a vehicle by hands-free voice command (or touch fallback) for risky or discourteous behavior
(`Weaving`, `Unsteady Throttle`, `Cutting`, `Tailgating`, `Speed`, `No Lane`, `No Plate`, `Asshole`)
or mark courteous drivers as **`★ Angel`**. When a previously recorded vehicle is recognized again —
via exact plate, partial/near plate + appearance, or obscured-plate appearance — Jolt raises a
severity-coded Tactical HUD (**Maroon / Red / Orange / Yellow / Angel Green**), haptic pattern, and
concise spoken TTS alert (*"Red. Weaver."*) while pausing Spotify/media automatically.

## How it works

The app watches the road through the phone's 5× telephoto lens, runs a locally-trained YOLOv8n
model on the GPU delegate to detect license plates, and reads foreground plates (`>=26px` wide)
with on-device OCR (Google ML Kit, bundled/offline) using multi-frame consensus voting, confusion
folding, and a **5-second forward plate backfill** for early highway voice flags.

When a plate can't be read — glare, angle, distance, or tinted cover — a highway-domain fine-tuned
EfficientNet-B0 GPU model marginalizes **Make** and **Body Style** (`SUV`, `Sedan`, `Pickup`, `Van`,
`Coupe`, `Wagon/Hatch`) across 196 Stanford Cars classes and fuses zero-copy HSV paint color
extraction to identify modern vehicle appearance (e.g. `"White Honda SUV"`). Hands-free voice
commands are spotted locally at 16 kHz via Vosk constrained grammar with zero cloud calls.

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
| ML inference | LiteRT (TFLite) with Dual GPU delegates, CPU fallback |
| Plate detection | Custom YOLOv8n (`yolov8_license_plate.tflite`, mAP50 = 97.2%) |
| OCR | Google ML Kit Text Recognition V2 (bundled, offline) + Tier 2 partial-fragment matcher |
| Vehicle appearance | Highway-tuned EfficientNet-B0 + marginalized Make/Body + HSV paint color |
| Voice & audio | Vosk Android (`vosk-model-small-en-us`, offline constrained grammar) + Android TTS with `AudioFocusRequest` (auto-pauses/resumes Spotify) |
| Database | Room (SQLite) v7, explicit versioned migrations (`MIGRATION_4_5`, `MIGRATION_5_6`, `MIGRATION_6_7`) |
| Analytics & Sync | Normalized encounter-rate Trends Dashboard (`White SUV`, `Honda/Acura/Infiniti`) + GPS-stripped Trusted-Group `.json` bundle share/import |
| Maps | OSMDroid (online street-tile download + local offline tile cache) |

## Building

1. Open Android Studio → **File → Open** → select this repo's `android/` folder (configured for JDK 21 LTS).
2. Let Gradle sync.
3. Connect a device (USB debugging enabled) and hit **Run**.

Command line:
```bash
cd android
./gradlew assembleDebug
```

**Note on model files:** The large offline ML & speech binaries are gitignored and live locally in `android/app/src/main/assets/`:
- `yolov8_license_plate.tflite` — custom YOLOv8n plate detector (~12 MB)
- `mmc_classifier.tflite` — highway-domain fine-tuned EfficientNet-B0 196-class vehicle model (~17 MB)
- `model-en-us/` — unpacked `vosk-model-small-en-us-0.15` directory (with `uuid` file) for offline `"Jolt..."` wake-phrase & command recognition

The app degrades gracefully if optional models (`mmc_classifier.tflite` or `model-en-us/`) are absent on a fresh clone.

## Hands-Free Voice Commands, Spotify Pause & Tactical HUD

Say **"Jolt"** (or **"Hey Jolt"**) while on the **Main** tab — Jolt automatically pauses Spotify / podcasts via `AudioFocusRequest` while listening and speaking, then resumes playback automatically:
- **Standalone wake + 8s command window:** Say `"Jolt"` → chime + yellow `🎙 LISTENING — SAY COMMAND…` badge → say `"weaving"`, `"cutting"`, `"angel"`, `"cancel"`, etc.
- **One-shot flagging:** `"Jolt, flag weaving"`, `"Jolt, flag white Honda SUV cutting and asshole"`, `"Jolt, flag tailgating"`
- **Two-step conversational flagging:** `"Jolt, flag that car"` → Jolt reads the target aloud (`"Plate L T 2 3 7 9. Reason?"`) → reply within 7 seconds (`"Line cutter"`, `"Unsteady throttle"`, `"No plate"`, etc.)
- **5-second forward plate backfill:** Flagging a distant vehicle slightly before its plate is readable automatically attaches the plate + crop if consensus locks within the next 5.5 seconds.
- **Positive Angel mark:** `"Jolt, angel that car"` (permanent green goodwill badge alongside any history)
- **Corrections, repeat & mute:** `"Jolt, wrong car"` / `"Jolt, cancel"`, `"Jolt, what was that"`, `"Jolt, mute alerts"` / `"Jolt, alerts on"`
- **History & Trends Tab:** Toggle between **Incidents** (search, filter, tap any card while parked to edit plate/color/make/body/tags/notes) and **Trends & Rates** (`White SUV` hypothesis spotlight, Central CT `Honda / Acura / Infiniti` watchlist, and normalized `flags ÷ encounters` rates).
- **Trusted-Group Share (Export tab):** Export or import GPS-stripped `.json` vehicle bundles across 3–4 family or commuter phones.
