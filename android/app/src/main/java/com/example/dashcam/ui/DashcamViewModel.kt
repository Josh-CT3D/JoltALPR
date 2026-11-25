package com.ct3d.jolt.ui

import android.Manifest
import android.annotation.SuppressLint
import android.app.Application
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.location.Location
import android.util.Log
import androidx.core.content.ContextCompat
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.ct3d.jolt.camera.PlateBox
import com.ct3d.jolt.camera.TelephotoAnalyzer
import com.ct3d.jolt.data.AppDatabase
import com.ct3d.jolt.data.BehaviorTag
import com.ct3d.jolt.data.DatabaseBackup
import com.ct3d.jolt.data.DriverLog
import com.ct3d.jolt.data.JoltSeverity
import com.ct3d.jolt.data.LocationRecord
import com.ct3d.jolt.data.VehicleReputation
import com.ct3d.jolt.data.computeVehicleReputation
import com.ct3d.jolt.data.normalizePlate
import com.ct3d.jolt.data.safeClearAllLocationRecords
import com.ct3d.jolt.data.safeClearAllLogs
import com.ct3d.jolt.data.safeDeleteLog
import com.ct3d.jolt.data.safeInsertLocationRecord
import com.ct3d.jolt.data.safeInsertLog
import com.ct3d.jolt.service.BatteryMonitor
import com.ct3d.jolt.service.FeedbackController
import com.ct3d.jolt.service.FieldTelemetryLogger
import com.ct3d.jolt.speech.VoiceCommand
import com.ct3d.jolt.speech.VoskVoiceController
import com.ct3d.jolt.training.TrainingDataCollector
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.google.android.gms.tasks.CancellationTokenSource
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import java.io.File
import kotlin.coroutines.resume

/**
 * MVVM ViewModel — manages ALPR detections, multi-incident Vehicle Reputation alerts,
 * Vosk hands-free wake-phrase commands, TTS spoken alerts, Room v6 persistence, GPS, and telemetry.
 */
class DashcamViewModel(application: Application) : AndroidViewModel(application) {

    private val db = AppDatabase.getDatabase(application)
    private val dao = db.driverLogDao()
    private val batteryMonitor = BatteryMonitor(application)
    private val trainingCollector = TrainingDataCollector(application)
    private val databaseBackup = DatabaseBackup(application)
    private val feedback = FeedbackController(application)
    private val telemetryLogger = FieldTelemetryLogger(application, batteryMonitor)

    private val voiceController = VoskVoiceController(application) { command, rawTranscript ->
        viewModelScope.launch {
            handleVoiceCommand(command, rawTranscript)
        }
    }

    val voiceState: StateFlow<VoskVoiceController.State> = voiceController.state

    // Debounce repeat encounters: one spoken alert + one auto-sighting per 30s cooldown window.
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

    private val _plateBoxes = MutableStateFlow<List<PlateBox>>(emptyList())
    val plateBoxes: StateFlow<List<PlateBox>> = _plateBoxes

    private val _frameAspectRatio = MutableStateFlow(0.75f)
    val frameAspectRatio: StateFlow<Float> = _frameAspectRatio

    // Sticky OCR (5s) + Sticky MMC (10s) so a flagged vehicle retains both plate AND Make/Model.
    private val _lastConfidentOcr = MutableStateFlow<String?>(null)
    val lastConfidentOcr: StateFlow<String?> = _lastConfidentOcr
    private var ocrStickyJob: Job? = null

    private var lastSeenMmc: String? = null
    private var mmcStickyJob: Job? = null

    // R4: ViewModel owns published plate crops on Main, expiring 5s after plates leave frame.
    private val _lastPlateCrop = MutableStateFlow<Bitmap?>(null)
    private var cropStickyJob: Job? = null

    private var dismissedAlertPlateNormalized: String? = null

    private val _notificationMessage = MutableStateFlow<String?>(null)
    val notificationMessage: StateFlow<String?> = _notificationMessage

    // Active quick-tag refinement state (shows 8 behavior chips for 7s after tapping FLAG)
    data class RecentFlagState(
        val rowId: Long,
        val vehicleLabel: String,
        val selectedTags: List<BehaviorTag> = emptyList()
    )
    private val _recentFlagState = MutableStateFlow<RecentFlagState?>(null)
    val recentFlagState: StateFlow<RecentFlagState?> = _recentFlagState
    private var recentFlagExpiryJob: Job? = null

    // Training mode (Phase 6)
    private val _trainingModeEnabled = MutableStateFlow(false)
    val trainingModeEnabled: StateFlow<Boolean> = _trainingModeEnabled

    private val _trainingSampleCount = MutableStateFlow(0)
    val trainingSampleCount: StateFlow<Int> = _trainingSampleCount

    // Field Telemetry & Soak Logger
    private val _telemetryEnabled = MutableStateFlow(telemetryLogger.isPersistedEnabled())
    val telemetryEnabled: StateFlow<Boolean> = _telemetryEnabled

    private val _telemetryEventCount = MutableStateFlow(telemetryLogger.currentEventCount())
    val telemetryEventCount: StateFlow<Int> = _telemetryEventCount
    private var healthTelemetryJob: Job? = null

    /**
     * Phase 2 Tactical Encounter Alert.
     * Non-null when a recognized plate has prior BAD or ANGEL history in Room DB.
     */
    private val _knownBadDriverAlert = MutableStateFlow<VehicleReputation?>(null)
    val knownBadDriverAlert: StateFlow<VehicleReputation?> = _knownBadDriverAlert

    // Declared BEFORE init (CLAUDE.md critical order #3)
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
        Log.i(TAG, "ViewModel initializing — starting GPS, TTS echo guard & telemetry.")
        feedback.onSpeechStateChanged = { speaking ->
            voiceController.setTtsSpeaking(speaking)
        }
        fetchUpdatedLocation()
        startLocationUpdates()
        trimOldBreadcrumbs()
        initializeVoiceIfPermitted()
        viewModelScope.launch {
            _trainingSampleCount.value = withContext(Dispatchers.IO) { trainingCollector.sampleCount() }
        }
        if (_telemetryEnabled.value) {
            startHealthTelemetryLoop(marker = "APP_RESUME_LOGGING")
        }
    }

    fun initializeVoiceIfPermitted() {
        val hasMic = ContextCompat.checkSelfPermission(
            getApplication(),
            Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
        if (hasMic) {
            voiceController.initialize()
        }
    }

    fun toggleVoiceMute() {
        val muted = voiceController.toggleUserMute()
        _notificationMessage.value = if (muted) "Voice wake-word muted" else "Voice wake-word active (say \"Jolt...\")"
    }

    private suspend fun handleVoiceCommand(command: VoiceCommand, rawTranscript: String) {
        if (_telemetryEnabled.value) {
            telemetryLogger.recordAction("VOICE_COMMAND", "cmd=${command::class.simpleName} raw=$rawTranscript", currentSpeedMph())
            _telemetryEventCount.value = telemetryLogger.currentEventCount()
        }

        when (command) {
            VoiceCommand.WakeDetected -> {
                feedback.wakeWordAcknowledged()
                _notificationMessage.value = "🎙 Listening — say \"flag\", \"weaving\", \"cutting\", \"angel\", or \"cancel\"…"
            }
            is VoiceCommand.FlagVehicle -> {
                flagBadDriver(
                    tags = command.tags,
                    severityOverride = command.severityOverride,
                    notes = command.descriptorNotes,
                    fromVoice = true
                )
            }
            is VoiceCommand.AngelVehicle -> {
                voiceController.closeFollowUpReasonWindow()
                flagAngelDriver(notes = command.descriptorNotes, fromVoice = true)
            }
            is VoiceCommand.FollowUpReason -> {
                voiceController.closeFollowUpReasonWindow()
                val activeRecent = _recentFlagState.value
                if (activeRecent != null) {
                    applyTagsToRecentFlag(
                        rowId = activeRecent.rowId,
                        tags = command.tags,
                        severityOverride = command.severityOverride,
                        speakFeedback = true
                    )
                } else {
                    flagBadDriver(
                        tags = command.tags,
                        severityOverride = command.severityOverride,
                        fromVoice = true
                    )
                }
            }
            VoiceCommand.UndoOrWrongCar -> {
                voiceController.closeFollowUpReasonWindow()
                undoLastAction(fromVoice = true)
            }
            VoiceCommand.MuteAlerts -> {
                voiceController.closeFollowUpReasonWindow()
                feedback.muteAlertsForTenMinutes()
                _notificationMessage.value = "Audio alerts muted for 10 minutes."
            }
            VoiceCommand.UnmuteAlerts -> {
                voiceController.closeFollowUpReasonWindow()
                feedback.unmuteAlerts()
                _notificationMessage.value = "Audio alerts unmuted."
            }
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

        if (vehicleMmc != null) {
            lastSeenMmc = vehicleMmc
            mmcStickyJob?.cancel()
            mmcStickyJob = viewModelScope.launch {
                delay(10_000L)
                lastSeenMmc = null
            }
        }

        // R4: ViewModel owns published plate crops on Main.
        if (plateCrop != null) {
            cropStickyJob?.cancel()
            cropStickyJob = null
            val prev = _lastPlateCrop.value
            _lastPlateCrop.value = plateCrop
            if (prev != null && prev !== plateCrop && !prev.isRecycled) prev.recycle()
        } else if (plateBoxes.isEmpty() && _lastPlateCrop.value != null && cropStickyJob == null) {
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
                    val reputation = lookupPlates.firstNotNullOfOrNull { token ->
                        val history = dao.findVehicleHistoryByPlate(normalizePlate(token))
                        computeVehicleReputation(history, matchConfidence = 1.0f)
                    }
                    if (reputation != null) {
                        if (reputation.plateNormalized != dismissedAlertPlateNormalized) {
                            _knownBadDriverAlert.value = reputation
                        }
                        val now = System.currentTimeMillis()
                        val newEncounter = reputation.plateNormalized != lastSightingPlate ||
                            now - lastSightingTime > SIGHTING_COOLDOWN_MS
                        if (newEncounter) {
                            lastSightingPlate = reputation.plateNormalized
                            lastSightingTime = now
                            feedback.announceEncounter(reputation)
                            if (reputation.badIncidentCount > 0) {
                                insertSighting(reputation, now)
                            }
                            if (_telemetryEnabled.value) {
                                telemetryLogger.recordAction(
                                    "ENCOUNTER_ALERT",
                                    "plate=${reputation.plateDisplay} sev=${reputation.effectiveSeverity.name} bad=${reputation.badIncidentCount} angel=${reputation.angelCount}",
                                    currentSpeedMph()
                                )
                                _telemetryEventCount.value = telemetryLogger.currentEventCount()
                            }
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Vehicle reputation lookup failed: ${e.localizedMessage}", e)
                }
            }
        }
    }

    private suspend fun insertSighting(reputation: VehicleReputation, timestamp: Long) {
        try {
            val loc = _currentLocation.value
            dao.safeInsertLog(
                DriverLog(
                    id = 0,
                    rating = "SIGHTING",
                    plateOcr = reputation.plateDisplay,
                    plateNormalized = reputation.plateNormalized,
                    vehicleMmc = reputation.vehicleMmc ?: lastSeenMmc,
                    timestamp = timestamp,
                    latitude = loc?.latitude ?: 0.0,
                    longitude = loc?.longitude ?: 0.0,
                    batteryLevel = batteryMonitor.getBatteryLevel(),
                    plateCropPath = null,
                    severity = reputation.effectiveSeverity.name
                )
            )
            Log.i(TAG, "Auto-logged sighting of known plate ${reputation.plateDisplay}")
        } catch (e: Exception) {
            Log.e(TAG, "Sighting insert failed: ${e.localizedMessage}", e)
        }
    }

    fun dismissAlert() {
        dismissedAlertPlateNormalized = _knownBadDriverAlert.value?.plateNormalized
        _knownBadDriverAlert.value = null
        Log.i(TAG, "Encounter alert dismissed.")
    }

    /**
     * Records a BAD incident via voice command or screen FLAG button.
     */
    fun flagBadDriver(
        tags: List<BehaviorTag> = emptyList(),
        severityOverride: JoltSeverity? = null,
        notes: String? = null,
        fromVoice: Boolean = false
    ) {
        feedback.flagPressed()
        val ocr = _lastConfidentOcr.value ?: _activePlateOcr.value
        val mmc = _activeVehicleMmc.value ?: lastSeenMmc
        val hasPlateContext = ocr != null || _plateBoxes.value.isNotEmpty()
        val cropCopy: Bitmap? = if (hasPlateContext) {
            _lastPlateCrop.value?.takeIf { !it.isRecycled }?.let { src ->
                try { src.copy(src.config ?: Bitmap.Config.ARGB_8888, false) } catch (_: Exception) { null }
            }
        } else null
        val cachedLocation = _currentLocation.value

        val cappedTags = tags.distinct().take(BehaviorTag.MAX_TAGS_PER_INCIDENT)
        val initialSeverity = BehaviorTag.defaultSeverityFor(cappedTags, severityOverride)

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
                    vehicleMmc = mmc, // Retained even when OCR succeeds!
                    timestamp = captureTime,
                    latitude = lat,
                    longitude = lon,
                    batteryLevel = battery,
                    plateCropPath = cropPath,
                    behaviorTags = BehaviorTag.serialize(cappedTags),
                    severity = initialSeverity.name,
                    matchConfidence = if (ocr != null) 1.0f else 0.65f,
                    notes = notes
                )

                val rowId = dao.safeInsertLog(newLog)
                val targetName = ocr ?: mmc ?: notes ?: "Unidentified Vehicle"
                val tagSuffix = if (cappedTags.isNotEmpty()) " (${cappedTags.joinToString { it.displayLabel }})" else ""
                _notificationMessage.value = "Flagged [${initialSeverity.displayLabel}]: $targetName$tagSuffix"
                Log.i(TAG, "Flagged bad driver [Row: $rowId, Target: $targetName, Sev: $initialSeverity, Tags: $cappedTags]")

                // Open 7-second quick-tag / voice follow-up window
                openRecentFlagWindow(rowId, targetName, cappedTags)

                if (fromVoice) {
                    val spokenTarget = when {
                        ocr != null -> "Plate ${feedback.formatPlateForSpeech(ocr)}"
                        mmc != null -> mmc
                        notes != null -> notes
                        else -> "vehicle"
                    }
                    if (cappedTags.isEmpty() && severityOverride == null) {
                        voiceController.openFollowUpReasonWindow(7_000L)
                        feedback.speak("$spokenTarget. Reason?")
                    } else {
                        voiceController.closeFollowUpReasonWindow()
                        val reasonSpoken = cappedTags.firstOrNull()?.spokenPhrase ?: initialSeverity.spokenLabel
                        feedback.speak("Flagged $spokenTarget. $reasonSpoken.")
                    }
                }

                if (_telemetryEnabled.value) {
                    telemetryLogger.recordAction(
                        if (fromVoice) "VOICE_FLAG" else "BUTTON_FLAG",
                        "target=$targetName sev=${initialSeverity.name} tags=${BehaviorTag.serialize(cappedTags) ?: "-"} crop=${cropPath != null}",
                        currentSpeedMph()
                    )
                    _telemetryEventCount.value = telemetryLogger.currentEventCount()
                }
            } catch (e: Exception) {
                if (cropCopy != null && !cropCopy.isRecycled) cropCopy.recycle()
                Log.e(TAG, "FLAG failed: ${e.localizedMessage}", e)
                _notificationMessage.value = "Save failed: ${e.localizedMessage}"
            }
        }
    }

    /**
     * Records a permanent positive Angel mark for courteous driving ("Jolt, angel that car" or ★ button).
     */
    fun flagAngelDriver(notes: String? = null, fromVoice: Boolean = false) {
        feedback.flagPressed()
        val ocr = _lastConfidentOcr.value ?: _activePlateOcr.value
        val mmc = _activeVehicleMmc.value ?: lastSeenMmc
        val hasPlateContext = ocr != null || _plateBoxes.value.isNotEmpty()
        val cropCopy: Bitmap? = if (hasPlateContext) {
            _lastPlateCrop.value?.takeIf { !it.isRecycled }?.let { src ->
                try { src.copy(src.config ?: Bitmap.Config.ARGB_8888, false) } catch (_: Exception) { null }
            }
        } else null
        val cachedLocation = _currentLocation.value

        viewModelScope.launch {
            try {
                val loc = resolveLocationForFlag() ?: cachedLocation
                val captureTime = System.currentTimeMillis()
                val cropPath = withContext(Dispatchers.IO) {
                    cropCopy?.let { bmp ->
                        try {
                            val cropsDir = File(getApplication<Application>().filesDir, "crops").also { it.mkdirs() }
                            val cropFile = File(cropsDir, "$captureTime.jpg")
                            cropFile.outputStream().use { out -> bmp.compress(Bitmap.CompressFormat.JPEG, 90, out) }
                            cropFile.absolutePath
                        } catch (_: Exception) {
                            null
                        } finally {
                            if (!bmp.isRecycled) bmp.recycle()
                        }
                    }
                }

                val angelLog = DriverLog(
                    id = 0,
                    rating = "ANGEL",
                    plateOcr = ocr,
                    plateNormalized = ocr?.let { normalizePlate(it) },
                    vehicleMmc = mmc,
                    timestamp = captureTime,
                    latitude = loc?.latitude ?: 0.0,
                    longitude = loc?.longitude ?: 0.0,
                    batteryLevel = batteryMonitor.getBatteryLevel(),
                    plateCropPath = cropPath,
                    severity = JoltSeverity.ANGEL.name,
                    matchConfidence = if (ocr != null) 1.0f else 0.65f,
                    notes = notes
                )
                dao.safeInsertLog(angelLog)
                val target = ocr ?: mmc ?: notes ?: "Vehicle"
                _notificationMessage.value = "★ Angel recorded: $target"
                if (fromVoice) {
                    val spoken = if (ocr != null) "Plate ${feedback.formatPlateForSpeech(ocr)}" else target
                    feedback.speak("Angel marked for $spoken.")
                }
            } catch (e: Exception) {
                if (cropCopy != null && !cropCopy.isRecycled) cropCopy.recycle()
                Log.e(TAG, "Angel save failed: ${e.localizedMessage}", e)
            }
        }
    }

    /** Toggle or set behavior tags on the just-flagged incident (up to 2 tags). */
    fun toggleQuickTagOnRecentFlag(tag: BehaviorTag) {
        val current = _recentFlagState.value ?: return
        val updated = if (tag in current.selectedTags) {
            current.selectedTags - tag
        } else {
            (current.selectedTags + tag).takeLast(BehaviorTag.MAX_TAGS_PER_INCIDENT)
        }
        viewModelScope.launch {
            applyTagsToRecentFlag(current.rowId, updated, severityOverride = null, speakFeedback = false)
        }
    }

    private suspend fun applyTagsToRecentFlag(
        rowId: Long,
        tags: List<BehaviorTag>,
        severityOverride: JoltSeverity?,
        speakFeedback: Boolean
    ) {
        val capped = tags.distinct().take(BehaviorTag.MAX_TAGS_PER_INCIDENT)
        val sev = BehaviorTag.defaultSeverityFor(capped, severityOverride)
        val serialized = BehaviorTag.serialize(capped)
        dao.updateLogTagsAndSeverity(rowId, serialized, sev.name)
        val label = _recentFlagState.value?.vehicleLabel ?: "Vehicle"
        _recentFlagState.value = RecentFlagState(rowId, label, capped)
        val tagNames = if (capped.isNotEmpty()) capped.joinToString { it.displayLabel } else sev.displayLabel
        _notificationMessage.value = "Updated [$label]: ${sev.displayLabel} • $tagNames"
        if (speakFeedback) {
            val spokenReason = capped.joinToString(" and ") { it.spokenPhrase }.ifEmpty { sev.spokenLabel }
            feedback.speak("Saved. ${sev.spokenLabel}, $spokenReason.")
        }
    }

    /**
     * Handles "Jolt, wrong car" / "Cancel" / Undo tap:
     *  1. If an encounter alert is active on screen and no flag was just created, suppresses the alert.
     *  2. Otherwise deletes the most recent user BAD/ANGEL flag (if created within the last 5 minutes).
     */
    fun undoLastAction(fromVoice: Boolean = false) {
        viewModelScope.launch {
            val recentFlag = _recentFlagState.value
            if (recentFlag == null && _knownBadDriverAlert.value != null) {
                dismissAlert()
                _notificationMessage.value = "Encounter alert suppressed."
                if (fromVoice) feedback.speak("Alert dismissed.")
                return@launch
            }

            val latest = dao.getLatestUserFlag()
            val now = System.currentTimeMillis()
            if (latest != null && now - latest.timestamp <= UNDO_MAX_AGE_MS) {
                deleteLogItem(latest)
                _recentFlagState.value = null
                recentFlagExpiryJob?.cancel()
                val label = latest.plateOcr ?: latest.vehicleMmc ?: "last report"
                _notificationMessage.value = "Removed report: $label"
                if (fromVoice) feedback.speak("Removed last report.")
            } else {
                _notificationMessage.value = "Nothing recent to undo."
                if (fromVoice) feedback.speak("Nothing recent to remove.")
            }
        }
    }

    private fun openRecentFlagWindow(rowId: Long, label: String, initialTags: List<BehaviorTag>) {
        _recentFlagState.value = RecentFlagState(rowId, label, initialTags)
        recentFlagExpiryJob?.cancel()
        recentFlagExpiryJob = viewModelScope.launch {
            delay(7_500L)
            _recentFlagState.value = null
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
        _notificationMessage.value = "Training Mode: $state"
    }

    fun onTrainingSample(frame: Bitmap, boxes: List<RectF>) {
        if (trainingCollector.onPlateDetected(frame, boxes)) {
            _trainingSampleCount.update { it + 1 }
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
        voiceController.setTabActive(tabName == "Main")
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
        } catch (e: SecurityException) {
            Log.e(TAG, "GPS permission denied — continuous location unavailable.", e)
        }
    }

    override fun onCleared() {
        super.onCleared()
        locationCancellationSource.cancel()
        ocrStickyJob?.cancel()
        mmcStickyJob?.cancel()
        cropStickyJob?.cancel()
        recentFlagExpiryJob?.cancel()
        healthTelemetryJob?.cancel()
        val crop = _lastPlateCrop.value
        _lastPlateCrop.value = null
        if (crop != null && !crop.isRecycled) crop.recycle()
        voiceController.release()
        feedback.release()
        telemetryLogger.release()
        try { fusedLocationClient.removeLocationUpdates(locationCallback) }
        catch (e: Exception) { Log.e(TAG, "Failed to remove location callback: ${e.localizedMessage}", e) }
    }

    companion object {
        private const val TAG = "DashcamViewModel"
        private const val SIGHTING_COOLDOWN_MS = 30_000L
        private const val HEALTH_SNAPSHOT_INTERVAL_MS = 15_000L
        private const val UNDO_MAX_AGE_MS = 5 * 60 * 1_000L
        private const val MPS_TO_MPH = 2.2369363f
    }
}
