package com.ct3d.jolt.ui

import android.annotation.SuppressLint
import android.app.Application
import android.graphics.Bitmap
import android.graphics.RectF
import android.location.Location
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.ct3d.jolt.data.AppDatabase
import com.ct3d.jolt.data.DatabaseBackup
import com.ct3d.jolt.data.DriverLog
import com.ct3d.jolt.data.LocationRecord
import com.ct3d.jolt.data.normalizePlate
import com.ct3d.jolt.camera.PlateBox
import com.ct3d.jolt.camera.TelephotoAnalyzer
import com.ct3d.jolt.data.safeInsertLog
import com.ct3d.jolt.data.safeDeleteLog
import com.ct3d.jolt.data.safeClearAllLogs
import com.ct3d.jolt.data.safeInsertLocationRecord
import com.ct3d.jolt.data.safeClearAllLocationRecords
import com.ct3d.jolt.service.BatteryMonitor
import com.ct3d.jolt.service.FeedbackController
import com.ct3d.jolt.service.FieldTelemetryLogger
import com.ct3d.jolt.training.TrainingDataCollector
import com.google.android.gms.location.*
import com.google.android.gms.tasks.CancellationTokenSource
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.io.File
import kotlin.coroutines.resume

/**
 * MVVM ViewModel — manages all app state, Room DB writes, GPS tracking,
 * and the active ALPR pipeline results.
 *
 * Key change from original: voice command handling replaced by flagBadDriver()
 * which is triggered by the physical FLAG button on the main screen.
 */
class DashcamViewModel(application: Application) : AndroidViewModel(application) {

    private val db = AppDatabase.getDatabase(application)
    private val dao = db.driverLogDao()
    private val batteryMonitor = BatteryMonitor(application)
    private val trainingCollector = TrainingDataCollector(application)
    private val databaseBackup = DatabaseBackup(application)
    private val feedback = FeedbackController(application)
    private val telemetryLogger = FieldTelemetryLogger(application, batteryMonitor)

    // Debounce known-bad encounters: the OCR pipeline reports the same plate many times/sec, but we
    // only want one alert buzz + one auto-sighting per encounter (or per cooldown window). (A21)
    private var lastSightingPlate: String? = null
    private var lastSightingTime: Long = 0L

    // Room Flows — all flagged capture records
    val logsList: StateFlow<List<DriverLog>> = dao.getAllLogs()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000L), emptyList())

    // GPS breadcrumb history
    val locationHistoryList: StateFlow<List<LocationRecord>> = dao.getAllLocationRecords()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000L), emptyList())

    private val fusedLocationClient: FusedLocationProviderClient =
        LocationServices.getFusedLocationProviderClient(application)

    private val _currentLocation = MutableStateFlow<Location?>(null)
    val currentLocation: StateFlow<Location?> = _currentLocation

    // Active ALPR pipeline results from TelephotoAnalyzer
    private val _activePlateOcr = MutableStateFlow<String?>(null)
    val activePlateOcr: StateFlow<String?> = _activePlateOcr

    private val _activeVehicleMmc = MutableStateFlow<String?>(null)
    val activeVehicleMmc: StateFlow<String?> = _activeVehicleMmc

    // Plate bounding boxes (fractional 0–1 coords) for the camera preview overlay
    private val _plateBoxes = MutableStateFlow<List<PlateBox>>(emptyList())
    val plateBoxes: StateFlow<List<PlateBox>> = _plateBoxes

    // Analyzed frame aspect ratio (width / height) for accurate FILL_CENTER overlay alignment.
    private val _frameAspectRatio = MutableStateFlow(0.75f)
    val frameAspectRatio: StateFlow<Float> = _frameAspectRatio

    // Sticky OCR: holds the last confident plate read for 5s after it leaves frame,
    // so the FLAG button always captures the most recently seen plate.
    private val _lastConfidentOcr = MutableStateFlow<String?>(null)
    val lastConfidentOcr: StateFlow<String?> = _lastConfidentOcr
    private var ocrStickyJob: Job? = null

    // Most recent plate crop bitmap — held up to 5s after a plate leaves view, then cleared
    // and recycled so a stale plate photo never attaches to a subsequent MMC-only vehicle flag.
    private val _lastPlateCrop = MutableStateFlow<Bitmap?>(null)
    private var cropStickyJob: Job? = null

    // Plate dismissed by user tap (✕) — stays suppressed while continuously in view, resets once
    // the plate stays out of view past the sticky window.
    private var dismissedAlertPlateNormalized: String? = null

    // Transient notification banners shown on the main screen
    private val _notificationMessage = MutableStateFlow<String?>(null)
    val notificationMessage: StateFlow<String?> = _notificationMessage

    // Training mode toggle — when true, TelephotoAnalyzer auto-saves annotated frames (Phase 6)
    private val _trainingModeEnabled = MutableStateFlow(false)
    val trainingModeEnabled: StateFlow<Boolean> = _trainingModeEnabled

    // Count of collected training samples on disk (drives the Export tab label).
    private val _trainingSampleCount = MutableStateFlow(0)
    val trainingSampleCount: StateFlow<Int> = _trainingSampleCount

    // Field Telemetry & Soak Logger (persisted across restarts so mounted drives stay recorded)
    private val _telemetryEnabled = MutableStateFlow(telemetryLogger.isPersistedEnabled())
    val telemetryEnabled: StateFlow<Boolean> = _telemetryEnabled

    private val _telemetryEventCount = MutableStateFlow(telemetryLogger.currentEventCount())
    val telemetryEventCount: StateFlow<Int> = _telemetryEventCount
    private var healthTelemetryJob: Job? = null

    /**
     * Known Bad Driver Alert.
     * Non-null when the current OCR plate matches a previously flagged BAD record in Room DB.
     * Drives the yellow pulsing border overlay on the main screen.
     */
    private val _knownBadDriverAlert = MutableStateFlow<DriverLog?>(null)
    val knownBadDriverAlert: StateFlow<DriverLog?> = _knownBadDriverAlert

    // Kept alive so we can cancel any in-flight getCurrentLocation request in onCleared().
    private var locationCancellationSource = CancellationTokenSource()

    private val locationCallback = object : LocationCallback() {
        override fun onLocationResult(locationResult: LocationResult) {
            val location = locationResult.lastLocation ?: return
            _currentLocation.value = location
            viewModelScope.launch {
                try {
                    dao.safeInsertLocationRecord(
                        LocationRecord(
                            timestamp = System.currentTimeMillis(),
                            latitude = location.latitude,
                            longitude = location.longitude,
                            speed = location.speed,
                            accuracy = location.accuracy
                        )
                    )
                } catch (e: Exception) {
                    Log.e(TAG, "GPS breadcrumb write failed: ${e.localizedMessage}", e)
                }
            }
        }
    }

    init {
        Log.i(TAG, "ViewModel initializing — starting GPS telemetry.")
        fetchUpdatedLocation()
        startLocationUpdates()
        trimOldBreadcrumbs()
        viewModelScope.launch {
            _trainingSampleCount.value = withContext(Dispatchers.IO) { trainingCollector.sampleCount() }
        }
        if (_telemetryEnabled.value) {
            startHealthTelemetryLoop(marker = "APP_RESUME_LOGGING")
        }
    }

    private fun currentSpeedMph(): Float? {
        val loc = _currentLocation.value ?: return null
        return if (loc.hasSpeed()) loc.speed * MPS_TO_MPH else null
    }

    private fun startHealthTelemetryLoop(marker: String) {
        healthTelemetryJob?.cancel()
        telemetryLogger.recordSessionMarker(marker, TelephotoAnalyzer.ANALYSIS_INTERVAL_MS, currentSpeedMph())
        _telemetryEventCount.value = telemetryLogger.currentEventCount()
        healthTelemetryJob = viewModelScope.launch {
            while (true) {
                delay(HEALTH_SNAPSHOT_INTERVAL_MS)
                telemetryLogger.recordHealthSnapshot(currentSpeedMph())
                _telemetryEventCount.value = telemetryLogger.currentEventCount()
            }
        }
    }

    private fun trimOldBreadcrumbs() {
        viewModelScope.launch {
            val cutoff = System.currentTimeMillis() - 24 * 60 * 60 * 1000L
            try {
                dao.deleteLocationRecordsOlderThan(cutoff)
            } catch (e: Exception) {
                Log.e(TAG, "Breadcrumb trim failed: ${e.localizedMessage}", e)
            }
        }
    }

    /**
     * Called by TelephotoAnalyzer via pipeline state collection in MainActivity.
     * Checks all confirmed plate tokens in the frame against Room DB so known-bad plates in
     * adjacent lanes still trigger alerts, and manages sticky crop / alert lifecycle cleanly.
     */
    fun updateActiveDetection(
        plateOcr: String?,
        vehicleMmc: String?,
        plateBoxes: List<PlateBox> = emptyList(),
        plateCrop: Bitmap? = null,
        confirmedPlates: List<String> = listOfNotNull(plateOcr),
        frameWidth: Int = 0,
        frameHeight: Int = 0
    ) {
        _activePlateOcr.value = plateOcr
        _activeVehicleMmc.value = vehicleMmc
        _plateBoxes.value = plateBoxes
        if (frameWidth > 0 && frameHeight > 0) {
            _frameAspectRatio.value = frameWidth.toFloat() / frameHeight.toFloat()
        }

        // R4: the ViewModel owns published plate crops on the main dispatcher.
        if (plateCrop != null) {
            cropStickyJob?.cancel()
            cropStickyJob = null
            val prev = _lastPlateCrop.value
            _lastPlateCrop.value = plateCrop
            if (prev != null && prev !== plateCrop && !prev.isRecycled) prev.recycle()
        } else if (plateBoxes.isEmpty() && _lastPlateCrop.value != null && cropStickyJob == null) {
            // Expire & recycle the held crop after 5s with no plate in view so it can never
            // attach to an unrelated vehicle flagged later via MMC fallback.
            cropStickyJob = viewModelScope.launch {
                delay(5_000L)
                val stale = _lastPlateCrop.value
                _lastPlateCrop.value = null
                if (stale != null && !stale.isRecycled) stale.recycle()
                cropStickyJob = null
            }
        }

        val lookupPlates = confirmedPlates.ifEmpty { listOfNotNull(plateOcr) }
        if (lookupPlates.isNotEmpty()) {
            val primary = plateOcr ?: lookupPlates.first()
            _lastConfidentOcr.value = primary
            ocrStickyJob?.cancel()
            ocrStickyJob = viewModelScope.launch {
                delay(5_000L)
                _lastConfidentOcr.value = null
                dismissedAlertPlateNormalized = null
            }

            viewModelScope.launch {
                try {
                    val match = lookupPlates.firstNotNullOfOrNull { token ->
                        dao.findBadDriverByPlate(normalizePlate(token))
                    }
                    if (match != null) {
                        // Do not re-open if the user explicitly tapped ✕ while continuously viewing
                        // this same vehicle.
                        if (match.plateNormalized != dismissedAlertPlateNormalized) {
                            _knownBadDriverAlert.value = match
                        }
                        Log.w(TAG, "⚠️ KNOWN BAD DRIVER DETECTED: ${match.plateOcr} (flagged at ${match.timestamp})")
                        val now = System.currentTimeMillis()
                        val newEncounter = match.plateNormalized != lastSightingPlate ||
                            now - lastSightingTime > SIGHTING_COOLDOWN_MS
                        if (newEncounter) {
                            lastSightingPlate = match.plateNormalized
                            lastSightingTime = now
                            feedback.badDriverAlert()
                            insertSighting(match, now)
                            if (_telemetryEnabled.value) {
                                telemetryLogger.recordAction(
                                    "BAD_DRIVER_ALERT",
                                    "plate=${match.plateOcr ?: "unknown"}",
                                    currentSpeedMph()
                                )
                                _telemetryEventCount.value = telemetryLogger.currentEventCount()
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Bad driver DB lookup failed: ${e.localizedMessage}", e)
                }
            }
        }
        // Note: when lookupPlates is empty (e.g. brief occlusion or plate leaves frame), do NOT
        // immediately clear _knownBadDriverAlert — let PulsingAlertBorder's 15s timer or ✕ tap
        // dismiss it cleanly.
    }

    /**
     * A21: log a lightweight "SIGHTING" row (no user action) when a known-bad plate is re-seen, so
     * the Map shows where repeat encounters happen. Rating "SIGHTING" (not "BAD") keeps these rows
     * out of findBadDriverByPlate lookups, so they never re-trigger alerts or spawn more sightings.
     */
    private suspend fun insertSighting(match: DriverLog, timestamp: Long) {
        try {
            val loc = _currentLocation.value
            dao.safeInsertLog(
                DriverLog(
                    id = 0,
                    rating = "SIGHTING",
                    plateOcr = match.plateOcr,
                    plateNormalized = match.plateNormalized,
                    vehicleMmc = null,
                    timestamp = timestamp,
                    latitude = loc?.latitude ?: 0.0,
                    longitude = loc?.longitude ?: 0.0,
                    batteryLevel = batteryMonitor.getBatteryLevel(),
                    plateCropPath = null
                )
            )
            Log.i(TAG, "Auto-logged sighting of known-bad plate ${match.plateOcr}")
        } catch (e: Exception) {
            Log.e(TAG, "Sighting insert failed: ${e.localizedMessage}", e)
        }
    }

    /** Manually dismisses the known bad driver alert (user taps ✕ or 15s auto-dismiss fires). */
    fun dismissAlert() {
        dismissedAlertPlateNormalized = _knownBadDriverAlert.value?.plateNormalized
        _knownBadDriverAlert.value = null
        Log.i(TAG, "Known bad driver alert dismissed.")
    }

    /**
     * Triggered when the FLAG BAD DRIVER button is pressed.
     * Snapshots current OCR/MMC and a safe copy of the plate crop on Main (preserving R4
     * single-thread ownership), then performs GPS lookup, JPEG compression, and Room insert off
     * the main thread.
     */
    fun flagBadDriver() {
        feedback.flagPressed()
        // Snapshot UI-thread-owned state synchronously before hopping to IO.
        val ocr = _lastConfidentOcr.value ?: _activePlateOcr.value
        val mmc = _activeVehicleMmc.value
        val hasPlateContext = ocr != null || _plateBoxes.value.isNotEmpty()
        // Copy the tiny plate crop on Main so Dispatchers.IO never races updateActiveDetection()'s recycle().
        val cropCopy: Bitmap? = if (hasPlateContext) {
            _lastPlateCrop.value?.takeIf { !it.isRecycled }?.let { src ->
                try { src.copy(src.config ?: Bitmap.Config.ARGB_8888, false) } catch (_: Exception) { null }
            }
        } else null
        val cachedLocation = _currentLocation.value

        viewModelScope.launch {
            try {
                val loc = resolveLocationForFlag() ?: cachedLocation
                val lat = loc?.latitude ?: 0.0
                val lon = loc?.longitude ?: 0.0
                val battery = batteryMonitor.getBatteryLevel()
                val captureTime = System.currentTimeMillis()

                val cropPath = withContext(Dispatchers.IO) {
                    cropCopy?.let { bmp ->
                        try {
                            val cropsDir = File(getApplication<Application>().filesDir, "crops").also { it.mkdirs() }
                            val cropFile = File(cropsDir, "$captureTime.jpg")
                            cropFile.outputStream().use { out -> bmp.compress(Bitmap.CompressFormat.JPEG, 90, out) }
                            Log.i(TAG, "Plate crop saved: ${cropFile.absolutePath}")
                            cropFile.absolutePath
                        } catch (e: Exception) {
                            Log.e(TAG, "Plate crop save failed: ${e.localizedMessage}", e)
                            null
                        } finally {
                            if (!bmp.isRecycled) bmp.recycle()
                        }
                    }
                }

                val newLog = DriverLog(
                    id = 0,
                    rating = "BAD",
                    plateOcr = ocr,
                    plateNormalized = ocr?.let { normalizePlate(it) },
                    vehicleMmc = if (ocr == null) mmc else null,
                    timestamp = captureTime,
                    latitude = lat,
                    longitude = lon,
                    batteryLevel = battery,
                    plateCropPath = cropPath
                )

                val rowId = dao.safeInsertLog(newLog)
                val tag = ocr ?: mmc ?: "Unidentified Vehicle"
                val latStr = String.format("%.5f", lat)
                val lonStr = String.format("%.5f", lon)
                _notificationMessage.value = "Flagged: $tag  |  GPS: [$latStr, $lonStr]  |  Bat: ${BatteryMonitor.format(battery)}"
                Log.i(TAG, "Flagged bad driver [Row: $rowId, Tag: $tag, GPS: $lat/$lon, Battery: $battery%]")
                if (_telemetryEnabled.value) {
                    telemetryLogger.recordAction(
                        "FLAG_PRESSED",
                        "tag=$tag crop=${cropPath != null} gps=$latStr,$lonStr",
                        currentSpeedMph()
                    )
                    _telemetryEventCount.value = telemetryLogger.currentEventCount()
                }
            } catch (e: Exception) {
                if (cropCopy != null && !cropCopy.isRecycled) cropCopy.recycle()
                Log.e(TAG, "FLAG BUTTON — DB write failed: ${e.localizedMessage}", e)
                _notificationMessage.value = "Save failed: ${e.localizedMessage}"
            }
        }
    }

    @SuppressLint("MissingPermission")
    private suspend fun resolveLocationForFlag(): Location? = withTimeoutOrNull(1_200L) {
        suspendCancellableCoroutine { cont ->
            locationCancellationSource.cancel()
            locationCancellationSource = CancellationTokenSource()
            try {
                fusedLocationClient.getCurrentLocation(
                    Priority.PRIORITY_HIGH_ACCURACY,
                    locationCancellationSource.token
                ).addOnSuccessListener { location ->
                    if (location != null) _currentLocation.value = location
                    if (cont.isActive) cont.resume(location)
                }.addOnFailureListener { e ->
                    Log.w(TAG, "Ad-hoc location query failed: ${e.localizedMessage}", e)
                    if (cont.isActive) cont.resume(null)
                }
            } catch (e: SecurityException) {
                Log.e(TAG, "Location permission missing.", e)
                if (cont.isActive) cont.resume(null)
            }
        }
    }

    fun toggleTrainingMode() {
        _trainingModeEnabled.value = !_trainingModeEnabled.value
        val state = if (_trainingModeEnabled.value) "ON" else "OFF"
        Log.i(TAG, "Training mode toggled: $state")
        _notificationMessage.value = "Training Mode: $state"
    }

    /**
     * Called from TelephotoAnalyzer's training callback (only when training mode is on) for each
     * high-confidence plate detection. Saves a frame + YOLO annotation. Runs on the analyzer's
     * pipeline thread and must finish before the frame bitmap is recycled — onPlateDetected saves
     * synchronously, so that holds.
     */
    fun onTrainingSample(frame: Bitmap, boxes: List<RectF>) {
        if (trainingCollector.onPlateDetected(frame, boxes)) {
            _trainingSampleCount.update { it + 1 } // R6: atomic — runs on the analyzer's pipeline thread
        }
    }

    fun clearTrainingData() {
        viewModelScope.launch {
            withContext(Dispatchers.IO) { trainingCollector.clearAll() }
            _trainingSampleCount.value = 0
            _notificationMessage.value = "Training samples cleared."
        }
    }

    fun toggleTelemetryLogging() {
        val next = !_telemetryEnabled.value
        _telemetryEnabled.value = next
        telemetryLogger.setPersistedEnabled(next)
        if (next) {
            startHealthTelemetryLoop(marker = "USER_START_LOGGING")
            _notificationMessage.value = "Field Telemetry Logging: ON"
        } else {
            healthTelemetryJob?.cancel()
            healthTelemetryJob = null
            telemetryLogger.recordSessionMarker("USER_STOP_LOGGING", TelephotoAnalyzer.ANALYSIS_INTERVAL_MS, currentSpeedMph())
            _telemetryEventCount.value = telemetryLogger.currentEventCount()
            _notificationMessage.value = "Field Telemetry Logging: OFF"
        }
    }

    fun onFrameTelemetry(frame: TelephotoAnalyzer.FrameTelemetry) {
        if (!_telemetryEnabled.value) return
        telemetryLogger.recordFrame(frame, currentSpeedMph())
        _telemetryEventCount.value = telemetryLogger.currentEventCount()
    }

    fun onTabSelected(tabName: String) {
        if (!_telemetryEnabled.value) return
        telemetryLogger.recordAction("TAB_SELECTED", "tab=$tabName", currentSpeedMph())
        _telemetryEventCount.value = telemetryLogger.currentEventCount()
    }

    fun exportTelemetryReport() {
        viewModelScope.launch {
            _notificationMessage.value = "Building telemetry report…"
            val fileName = withContext(Dispatchers.IO) { telemetryLogger.exportReportToDownloads() }
            _notificationMessage.value = if (fileName != null) {
                "Telemetry saved: Downloads/$fileName"
            } else {
                "No telemetry recorded yet."
            }
        }
    }

    fun clearTelemetryLogs() {
        viewModelScope.launch {
            withContext(Dispatchers.IO) { telemetryLogger.clearAll() }
            _telemetryEventCount.value = 0
            _notificationMessage.value = "Telemetry logs cleared."
        }
    }

    /**
     * A21 #4: back up the Room DB + evidence crops to Downloads/ as a ZIP.
     * Guards against device loss/wipe — migrations only protect against schema bumps.
     */
    fun backupDatabase() {
        viewModelScope.launch {
            _notificationMessage.value = "Backing up records…"
            val name = withContext(Dispatchers.IO) { databaseBackup.exportBackupToDownloads(db) }
            _notificationMessage.value = if (name != null) {
                "Backup saved: Downloads/$name"
            } else {
                "Backup failed — see logs."
            }
        }
    }

    /** Zips collected training samples to Downloads/ (Roboflow/YOLO layout) for Google Drive sync. */
    fun exportTrainingData() {
        viewModelScope.launch {
            val zipName = withContext(Dispatchers.IO) { trainingCollector.exportZipToDownloads() }
            _notificationMessage.value = if (zipName != null) {
                "Training data exported: Downloads/$zipName"
            } else {
                "No training data to export yet."
            }
        }
    }

    fun deleteLogItem(log: DriverLog) {
        viewModelScope.launch {
            try {
                dao.safeDeleteLog(log)
                log.plateCropPath?.let { path ->
                    withContext(Dispatchers.IO) {
                        try { File(path).delete() } catch (_: Exception) {}
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Delete log ${log.id} failed: ${e.localizedMessage}", e)
            }
        }
    }

    fun clearLogs() {
        viewModelScope.launch {
            try {
                dao.safeClearAllLogs()
                dao.safeClearAllLocationRecords()
                withContext(Dispatchers.IO) {
                    File(getApplication<Application>().filesDir, "crops").deleteRecursively()
                }
                _notificationMessage.value = "All records cleared."
            } catch (e: Exception) {
                Log.e(TAG, "Clear all failed: ${e.localizedMessage}", e)
            }
        }
    }

    fun dismissNotification() { _notificationMessage.value = null }

    @SuppressLint("MissingPermission")
    fun fetchUpdatedLocation() {
        locationCancellationSource.cancel()
        locationCancellationSource = CancellationTokenSource()
        try {
            fusedLocationClient.getCurrentLocation(
                Priority.PRIORITY_HIGH_ACCURACY,
                locationCancellationSource.token
            ).addOnSuccessListener { location ->
                if (location != null) _currentLocation.value = location
            }.addOnFailureListener { e ->
                Log.w(TAG, "Ad-hoc location query failed: ${e.localizedMessage}", e)
            }
        } catch (e: SecurityException) {
            Log.e(TAG, "Location permission missing.", e)
        }
    }

    @SuppressLint("MissingPermission")
    private fun startLocationUpdates() {
        try {
            val request = LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 5000L)
                .setMinUpdateIntervalMillis(2000L)
                .build()
            fusedLocationClient.requestLocationUpdates(
                request, locationCallback, android.os.Looper.getMainLooper()
            )
            Log.i(TAG, "Continuous GPS updates registered (5s interval).")
        } catch (e: SecurityException) {
            Log.e(TAG, "GPS permission denied — continuous location unavailable.", e)
        }
    }

    override fun onCleared() {
        super.onCleared()
        locationCancellationSource.cancel()
        ocrStickyJob?.cancel()
        cropStickyJob?.cancel()
        healthTelemetryJob?.cancel()
        val crop = _lastPlateCrop.value
        _lastPlateCrop.value = null
        if (crop != null && !crop.isRecycled) crop.recycle()
        feedback.release()
        telemetryLogger.release()
        try { fusedLocationClient.removeLocationUpdates(locationCallback) }
        catch (e: Exception) { Log.e(TAG, "Failed to remove location callback: ${e.localizedMessage}", e) }
    }

    companion object {
        private const val TAG = "DashcamViewModel"

        // Same known-bad plate within this window counts as one encounter (one buzz + one sighting).
        private const val SIGHTING_COOLDOWN_MS = 30_000L
        private const val HEALTH_SNAPSHOT_INTERVAL_MS = 15_000L
        private const val MPS_TO_MPH = 2.2369363f
    }
}
