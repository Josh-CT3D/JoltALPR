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
import com.ct3d.jolt.data.DriverLog
import com.ct3d.jolt.data.LocationRecord
import com.ct3d.jolt.data.safeInsertLog
import com.ct3d.jolt.data.safeDeleteLog
import com.ct3d.jolt.data.safeClearAllLogs
import com.ct3d.jolt.data.safeInsertLocationRecord
import com.ct3d.jolt.data.safeClearAllLocationRecords
import com.ct3d.jolt.service.BatteryMonitor
import com.google.android.gms.location.*
import com.google.android.gms.tasks.CancellationTokenSource
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.io.File

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
    private val _plateBoxes = MutableStateFlow<List<RectF>>(emptyList())
    val plateBoxes: StateFlow<List<RectF>> = _plateBoxes

    // Sticky OCR: holds the last confident plate read for 5s after it leaves frame,
    // so the FLAG button always captures the most recently seen plate.
    private val _lastConfidentOcr = MutableStateFlow<String?>(null)
    val lastConfidentOcr: StateFlow<String?> = _lastConfidentOcr
    private var ocrStickyJob: Job? = null

    // Most recent plate crop bitmap — used to save a JPEG when FLAG is pressed.
    private val _lastPlateCrop = MutableStateFlow<Bitmap?>(null)

    // Transient notification banners shown on the main screen
    private val _notificationMessage = MutableStateFlow<String?>(null)
    val notificationMessage: StateFlow<String?> = _notificationMessage

    // Training mode toggle — when true, TelephotoAnalyzer auto-saves annotated frames (Phase 6)
    private val _trainingModeEnabled = MutableStateFlow(false)
    val trainingModeEnabled: StateFlow<Boolean> = _trainingModeEnabled

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
     * When a new OCR plate is received, immediately queries Room to check if
     * this plate was ever flagged as a bad driver — triggers the alert if so.
     */
    fun updateActiveDetection(
        plateOcr: String?,
        vehicleMmc: String?,
        plateBoxes: List<RectF> = emptyList(),
        plateCrop: Bitmap? = null
    ) {
        _activePlateOcr.value = plateOcr
        _activeVehicleMmc.value = vehicleMmc
        _plateBoxes.value = plateBoxes
        if (plateCrop != null) _lastPlateCrop.value = plateCrop

        if (plateOcr != null) {
            // Update sticky OCR and reset the 5-second clear timer
            _lastConfidentOcr.value = plateOcr
            ocrStickyJob?.cancel()
            ocrStickyJob = viewModelScope.launch {
                delay(5_000L)
                _lastConfidentOcr.value = null
            }

            viewModelScope.launch {
                try {
                    val match = dao.findBadDriverByPlate(plateOcr)
                    _knownBadDriverAlert.value = match
                    if (match != null) {
                        Log.w(TAG, "⚠️ KNOWN BAD DRIVER DETECTED: $plateOcr (previously flagged at ${match.timestamp})")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Bad driver DB lookup failed: ${e.localizedMessage}", e)
                }
            }
        } else {
            // No plate in view — clear any active alert (sticky OCR timer handles its own clear)
            _knownBadDriverAlert.value = null
        }
    }

    /** Manually dismisses the known bad driver alert (user taps ✕ or 15s auto-dismiss fires). */
    fun dismissAlert() {
        _knownBadDriverAlert.value = null
        Log.i(TAG, "Known bad driver alert dismissed by user.")
    }

    /**
     * Triggered when the FLAG BAD DRIVER button is pressed.
     * Captures current GPS, active OCR/MMC, battery level, and writes a Room record.
     * Plate crop image saving will be added in Phase 3.
     */
    fun flagBadDriver() {
        viewModelScope.launch {
            try {
                fetchUpdatedLocation()
                val loc = _currentLocation.value
                val lat = loc?.latitude ?: 0.0
                val lon = loc?.longitude ?: 0.0
                // Use sticky OCR (holds 5s after plate leaves frame) so flag always captures last read plate
                val ocr = _lastConfidentOcr.value ?: _activePlateOcr.value
                val mmc = _activeVehicleMmc.value
                val battery = batteryMonitor.getBatteryLevel()

                // Save plate crop JPEG to filesDir/crops/<timestamp>.jpg
                val cropPath = _lastPlateCrop.value?.let { bmp ->
                    try {
                        val cropsDir = File(getApplication<Application>().filesDir, "crops").also { it.mkdirs() }
                        val cropFile = File(cropsDir, "${System.currentTimeMillis()}.jpg")
                        cropFile.outputStream().use { out -> bmp.compress(Bitmap.CompressFormat.JPEG, 90, out) }
                        Log.i(TAG, "Plate crop saved: ${cropFile.absolutePath}")
                        cropFile.absolutePath
                    } catch (e: Exception) {
                        Log.e(TAG, "Plate crop save failed: ${e.localizedMessage}", e)
                        null
                    }
                }

                val newLog = DriverLog(
                    id = 0,
                    rating = "BAD",
                    plateOcr = ocr,
                    vehicleMmc = if (ocr == null) mmc else null,
                    timestamp = System.currentTimeMillis(),
                    latitude = lat,
                    longitude = lon,
                    batteryLevel = battery,
                    plateCropPath = cropPath
                )

                val rowId = dao.safeInsertLog(newLog)
                val tag = ocr ?: mmc ?: "Unidentified Vehicle"
                val latStr = String.format("%.5f", lat)
                val lonStr = String.format("%.5f", lon)
                _notificationMessage.value = "Flagged: $tag  |  GPS: [$latStr, $lonStr]  |  Bat: $battery%"
                Log.i(TAG, "Flagged bad driver [Row: $rowId, Tag: $tag, GPS: $lat/$lon, Battery: $battery%]")
            } catch (e: Exception) {
                Log.e(TAG, "FLAG BUTTON — DB write failed: ${e.localizedMessage}", e)
                _notificationMessage.value = "Save failed: ${e.localizedMessage}"
            }
        }
    }

    fun toggleTrainingMode() {
        _trainingModeEnabled.value = !_trainingModeEnabled.value
        val state = if (_trainingModeEnabled.value) "ON" else "OFF"
        Log.i(TAG, "Training mode toggled: $state")
        _notificationMessage.value = "Training Mode: $state"
    }

    fun deleteLogItem(log: DriverLog) {
        viewModelScope.launch {
            try { dao.safeDeleteLog(log) }
            catch (e: Exception) { Log.e(TAG, "Delete log ${log.id} failed: ${e.localizedMessage}", e) }
        }
    }

    fun clearLogs() {
        viewModelScope.launch {
            try {
                dao.safeClearAllLogs()
                dao.safeClearAllLocationRecords()
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
        try { fusedLocationClient.removeLocationUpdates(locationCallback) }
        catch (e: Exception) { Log.e(TAG, "Failed to remove location callback: ${e.localizedMessage}", e) }
    }

    companion object {
        private const val TAG = "DashcamViewModel"
    }
}
