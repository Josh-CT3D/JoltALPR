package com.ct3d.jolt

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.hardware.camera2.CameraCharacteristics
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.util.LruCache
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size as GeoSize
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.lifecycleScope
import com.ct3d.jolt.camera.PlateBox
import com.ct3d.jolt.camera.TelephotoAnalyzer
import com.ct3d.jolt.data.BehaviorTag
import com.ct3d.jolt.data.DriverLog
import com.ct3d.jolt.data.JoltSeverity
import com.ct3d.jolt.data.PartialMatchEngine
import com.ct3d.jolt.data.TrendsAnalyticsEngine
import com.ct3d.jolt.data.VehicleReputation
import com.ct3d.jolt.speech.VoskVoiceController
import com.ct3d.jolt.ui.DashcamViewModel
import com.ct3d.jolt.ui.theme.Dimens
import com.ct3d.jolt.ui.theme.JoltColors
import com.ct3d.jolt.ui.theme.JoltTheme
import com.google.common.util.concurrent.ListenableFuture
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Main Activity — Jolt ALPR Road-Neighbor Awareness System (Phase 2.1 + Phase 3).
 */
class MainActivity : ComponentActivity() {

    private val viewModel: DashcamViewModel by viewModels()
    private lateinit var cameraExecutor: ExecutorService

    // R12: Compose state, NOT a plain var (DO NOT REVERT).
    private var telephotoAnalyzer by mutableStateOf<TelephotoAnalyzer?>(null)

    private val requiredPermissions = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.ACCESS_COARSE_LOCATION,
        Manifest.permission.RECORD_AUDIO
    )

    private val permissionsLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val cameraOk   = permissions[Manifest.permission.CAMERA] ?: false
        val locationOk = permissions[Manifest.permission.ACCESS_FINE_LOCATION] ?: false
        if (cameraOk && locationOk) {
            setupServices()
            viewModel.initializeVoiceIfPermitted()
        } else {
            Toast.makeText(this, "Camera and GPS permissions are required.", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        cameraExecutor = Executors.newSingleThreadExecutor()

        Configuration.getInstance().apply {
            userAgentValue = applicationContext.packageName
            load(
                applicationContext,
                applicationContext.getSharedPreferences("osmdroid", Context.MODE_PRIVATE)
            )
        }

        if (corePermissionsGranted()) {
            setupServices()
            if (!allPermissionsGranted()) {
                permissionsLauncher.launch(requiredPermissions)
            }
        } else {
            permissionsLauncher.launch(requiredPermissions)
        }

        setContent {
            JoltTheme {
                JoltApp(
                    viewModel = viewModel,
                    getAnalyzer = { telephotoAnalyzer }
                )
            }
        }
    }

    private fun corePermissionsGranted() = listOf(
        Manifest.permission.CAMERA,
        Manifest.permission.ACCESS_FINE_LOCATION
    ).all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun allPermissionsGranted() = requiredPermissions.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun setupServices() {
        if (telephotoAnalyzer != null) return
        Log.i(TAG, "Initializing ALPR pipeline analyzer...")
        try {
            telephotoAnalyzer = TelephotoAnalyzer(applicationContext, lifecycleScope).also { analyzer ->
                analyzer.trainingCallback = { frame, boxes ->
                    if (viewModel.trainingModeEnabled.value) viewModel.onTrainingSample(frame, boxes)
                }
                analyzer.frameTelemetryCallback = { frameTelemetry ->
                    if (viewModel.telemetryEnabled.value) viewModel.onFrameTelemetry(frameTelemetry)
                }
                lifecycleScope.launch {
                    analyzer.pipelineState.collect { state ->
                        viewModel.updateActiveDetection(
                            plateOcr         = state.activePlateocr,
                            vehicleMmc       = state.activeVehicleMmc,
                            plateBoxes       = state.plateBoxes,
                            plateCrop        = state.lastPlateCrop,
                            confirmedPlates  = state.confirmedPlateTokens,
                            frameWidth       = state.frameWidth,
                            frameHeight      = state.frameHeight,
                            activeAppearance = state.activeAppearance,
                            partialPlateHint = state.partialPlateHint
                        )
                    }
                }
            }
            Log.i(TAG, "TelephotoAnalyzer ready.")
        } catch (e: Exception) {
            Log.e(TAG, "TelephotoAnalyzer setup failed: ${e.localizedMessage}", e)
            Toast.makeText(this, "ML Engine error: ${e.localizedMessage}", Toast.LENGTH_LONG).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        telephotoAnalyzer?.cleanup()
        cameraExecutor.shutdown()
    }

    companion object { private const val TAG = "MainActivity" }
}

// ─────────────────────────────────────────────────────────────────────────────
// Root app with bottom navigation
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun JoltApp(
    viewModel: DashcamViewModel,
    getAnalyzer: () -> TelephotoAnalyzer?
) {
    var selectedTab by remember { mutableIntStateOf(0) }

    val activeOcr          by viewModel.activePlateOcr.collectAsState()
    val activeMmc          by viewModel.activeVehicleMmc.collectAsState()
    val plateBoxes         by viewModel.plateBoxes.collectAsState()
    val frameAspectRatio   by viewModel.frameAspectRatio.collectAsState()
    val logList            by viewModel.logsList.collectAsState()
    val trendsReport       by viewModel.trendsReport.collectAsState()
    val gpsLocation        by viewModel.currentLocation.collectAsState()
    val bannerNotification by viewModel.notificationMessage.collectAsState()
    val badDriverAlert     by viewModel.knownBadDriverAlert.collectAsState()
    val voiceState         by viewModel.voiceState.collectAsState()
    val recentFlagState    by viewModel.recentFlagState.collectAsState()
    val trainingMode       by viewModel.trainingModeEnabled.collectAsState()
    val trainingCount      by viewModel.trainingSampleCount.collectAsState()
    val telemetryEnabled   by viewModel.telemetryEnabled.collectAsState()
    val telemetryCount     by viewModel.telemetryEventCount.collectAsState()

    val selectTab: (Int, String) -> Unit = { idx, name ->
        if (selectedTab != idx) {
            selectedTab = idx
            viewModel.onTabSelected(name)
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        Scaffold(
            bottomBar = {
                NavigationBar(
                    containerColor = JoltColors.navBar,
                    tonalElevation = 0.dp,
                    modifier = Modifier.height(80.dp)
                ) {
                    JoltNavItem(0, selectedTab, { selectTab(0, "Main") },    Icons.Default.Videocam,     "Main")
                    JoltNavItem(1, selectedTab, { selectTab(1, "Map") },     Icons.Default.Map,          "Map")
                    JoltNavItem(2, selectedTab, { selectTab(2, "History") }, Icons.Default.Assessment,   "History")
                    JoltNavItem(3, selectedTab, { selectTab(3, "Export") },  Icons.Default.FileDownload, "Export")
                    JoltNavItem(4, selectedTab, { selectTab(4, "Manage") },  Icons.Default.ManageSearch, "Manage")
                }
            }
        ) { paddingValues ->
            Box(
                modifier = Modifier
                    .padding(paddingValues)
                    .fillMaxSize()
                    .background(MaterialTheme.colorScheme.background)
            ) {
                when (selectedTab) {
                    0 -> MainScreen(
                        activeOcr             = activeOcr,
                        activeMmc             = activeMmc,
                        plateBoxes            = plateBoxes,
                        frameAspectRatio      = frameAspectRatio,
                        voiceState            = voiceState,
                        onToggleVoiceMute     = viewModel::toggleVoiceMute,
                        recentFlagState       = recentFlagState,
                        onToggleQuickTag      = viewModel::toggleQuickTagOnRecentFlag,
                        onUndoRecentFlag      = { viewModel.undoLastAction(fromVoice = false) },
                        telemetryRecording    = telemetryEnabled,
                        bannerNotification    = bannerNotification,
                        onDismissNotification = viewModel::dismissNotification,
                        onFlagBadDriver       = { viewModel.flagBadDriver() },
                        onFlagAngel           = { viewModel.flagAngelDriver() },
                        getAnalyzer           = getAnalyzer
                    )
                    1 -> JoltMapScreen(
                        logs            = logList,
                        currentLocation = gpsLocation
                    )
                    2 -> HistoryAndTrendsScreen(
                        logList        = logList,
                        trendsReport   = trendsReport,
                        onSaveEdits    = viewModel::saveIncidentReviewEdits,
                        onDeleteRecord = viewModel::deleteLogItem
                    )
                    3 -> ExportScreen(
                        logList             = logList,
                        trustedUnitId       = viewModel.trustedUnitId,
                        onExportGroupBundle = viewModel::exportTrustedGroupBundle,
                        onImportGroupBundle = viewModel::importTrustedGroupBundle,
                        trainingMode        = trainingMode,
                        trainingCount       = trainingCount,
                        onToggleTraining    = viewModel::toggleTrainingMode,
                        onExportTraining    = viewModel::exportTrainingData,
                        onClearTraining     = viewModel::clearTrainingData,
                        telemetryEnabled    = telemetryEnabled,
                        telemetryCount      = telemetryCount,
                        onToggleTelemetry   = viewModel::toggleTelemetryLogging,
                        onExportTelemetry   = viewModel::exportTelemetryReport,
                        onClearTelemetry    = viewModel::clearTelemetryLogs,
                        onBackupDatabase    = viewModel::backupDatabase
                    )
                    4 -> ManageScreen(
                        logList     = logList,
                        onDeleteLog = viewModel::deleteLogItem,
                        onClearAll  = viewModel::clearLogs
                    )
                }
            }
        }

        val activeAlert = badDriverAlert
        if (activeAlert != null) {
            TacticalEncounterHud(
                reputation = activeAlert,
                onDismiss  = viewModel::dismissAlert
            )
        }
    }
}

@Composable
private fun severityUiColor(severity: JoltSeverity): Color = when (severity) {
    JoltSeverity.MAROON -> JoltColors.severityMaroon
    JoltSeverity.RED    -> JoltColors.severityRed
    JoltSeverity.ORANGE -> JoltColors.severityOrange
    JoltSeverity.YELLOW -> JoltColors.severityYellow
    JoltSeverity.ANGEL  -> JoltColors.angelGreen
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 2 + Phase 3 Tactical Encounter HUD (Severity + Pips + Certainty)
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun TacticalEncounterHud(reputation: VehicleReputation, onDismiss: () -> Unit) {
    LaunchedEffect(reputation.plateNormalized, reputation.latestIncidentTimestamp) {
        kotlinx.coroutines.delay(15_000L)
        onDismiss()
    }

    val hudColor = severityUiColor(reputation.effectiveSeverity)
    val onHudTextColor = when (reputation.effectiveSeverity) {
        JoltSeverity.MAROON, JoltSeverity.RED -> Color.White
        else -> Color.Black
    }

    val isSuspected = reputation.matchConfidence < 0.95f
    val infiniteTransition = rememberInfiniteTransition(label = "alert_pulse")
    val borderAlpha by infiniteTransition.animateFloat(
        initialValue = if (isSuspected) 0.22f else 0.35f,
        targetValue  = if (isSuspected) 0.78f else 1.0f,
        animationSpec = infiniteRepeatable(
            animation  = tween(450, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "border_alpha"
    )

    Box(modifier = Modifier.fillMaxSize()) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .border(Dimens.alertBorder, hudColor.copy(alpha = borderAlpha))
        )

        Card(
            shape    = RoundedCornerShape(10.dp),
            colors   = CardDefaults.cardColors(containerColor = hudColor),
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = 8.dp, start = 12.dp, end = 12.dp)
        ) {
            Row(
                modifier          = Modifier.padding(horizontal = 14.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = when {
                        reputation.isPureAngel -> Icons.Default.Star
                        isSuspected -> Icons.Default.HelpOutline
                        else -> Icons.Default.Warning
                    },
                    contentDescription = null,
                    tint = onHudTextColor,
                    modifier = Modifier.size(22.dp)
                )
                Spacer(modifier = Modifier.width(10.dp))
                Column(modifier = Modifier.weight(1f, fill = false)) {
                    val pips = if (reputation.badIncidentCount > 0) {
                        val dots = "●".repeat(reputation.badIncidentCount.coerceAtMost(5))
                        val extra = if (reputation.badIncidentCount > 5) "+${reputation.badIncidentCount - 5}" else ""
                        "  $dots$extra"
                    } else ""
                    val halo = if (reputation.hasAngelHalo && !reputation.isPureAngel) "  ★ ANGEL" else ""
                    val qMark = if (isSuspected) " ?" else ""
                    Text(
                        text       = "${reputation.effectiveSeverity.displayLabel} • ${reputation.primaryHudBehavior}$pips$halo$qMark",
                        color      = onHudTextColor,
                        fontWeight = FontWeight.ExtraBold,
                        fontSize   = 13.sp,
                        fontFamily = FontFamily.Monospace
                    )
                    val certaintyLabel = if (!isSuspected) {
                        "CONFIRMED: ${reputation.plateDisplay}"
                    } else {
                        "${(reputation.matchConfidence * 100).toInt()}% SUSPECTED: ${reputation.plateDisplay}"
                    }
                    Text(
                        text       = "$certaintyLabel  •  ${reputation.effectiveSeverity.defensiveAction}",
                        color      = onHudTextColor.copy(alpha = 0.88f),
                        fontWeight = FontWeight.SemiBold,
                        fontSize   = 10.sp,
                        fontFamily = FontFamily.Monospace
                    )
                }
                Spacer(modifier = Modifier.width(8.dp))
                IconButton(onClick = onDismiss, modifier = Modifier.size(26.dp)) {
                    Icon(Icons.Default.Close, contentDescription = "Dismiss", tint = onHudTextColor)
                }
            }
        }
    }
}

@Composable
private fun RowScope.JoltNavItem(
    index: Int,
    selectedTab: Int,
    onClick: () -> Unit,
    icon: ImageVector,
    label: String
) {
    NavigationBarItem(
        selected = selectedTab == index,
        onClick  = onClick,
        icon     = { Icon(icon, contentDescription = label, modifier = Modifier.size(28.dp)) },
        label    = { Text(label, fontSize = 12.sp, fontWeight = FontWeight.SemiBold) },
        colors   = NavigationBarItemDefaults.colors(
            selectedIconColor   = MaterialTheme.colorScheme.primary,
            selectedTextColor   = MaterialTheme.colorScheme.primary,
            unselectedIconColor = Color.Gray,
            unselectedTextColor = Color.Gray,
            indicatorColor      = JoltColors.navIndicator
        )
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Screen
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun MainScreen(
    activeOcr: String?,
    activeMmc: String?,
    plateBoxes: List<PlateBox>,
    frameAspectRatio: Float,
    voiceState: VoskVoiceController.State,
    onToggleVoiceMute: () -> Unit,
    recentFlagState: DashcamViewModel.RecentFlagState?,
    onToggleQuickTag: (BehaviorTag) -> Unit,
    onUndoRecentFlag: () -> Unit,
    telemetryRecording: Boolean,
    bannerNotification: String?,
    onDismissNotification: () -> Unit,
    onFlagBadDriver: () -> Unit,
    onFlagAngel: () -> Unit,
    getAnalyzer: () -> TelephotoAnalyzer?
) {
    val context        = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    Column(modifier = Modifier.fillMaxSize()) {

        AnimatedVisibility(visible = bannerNotification != null) {
            Surface(
                color = MaterialTheme.colorScheme.tertiary,
                modifier = Modifier.fillMaxWidth()
            ) {
                Row(
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text       = bannerNotification ?: "",
                        color      = Color.Black,
                        fontSize   = 12.sp,
                        fontWeight = FontWeight.Medium,
                        modifier   = Modifier.weight(1f)
                    )
                    IconButton(onClick = onDismissNotification, modifier = Modifier.size(28.dp)) {
                        Icon(Icons.Default.Close, contentDescription = "Dismiss", tint = Color.Black)
                    }
                }
            }
        }

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .weight(4f)
                .background(Color.Black)
        ) {
            CameraPreviewWidget(
                lifecycleOwner       = lifecycleOwner,
                cameraProviderFuture = ProcessCameraProvider.getInstance(context),
                analyzer             = getAnalyzer()
            )

            BoundingBoxOverlay(
                boxes            = plateBoxes,
                frameAspectRatio = frameAspectRatio,
                modifier         = Modifier.fillMaxSize()
            )

            Row(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(8.dp),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .background(Color.Black.copy(alpha = 0.65f), RoundedCornerShape(4.dp))
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text(
                        text       = "5× TELEPHOTO",
                        color      = MaterialTheme.colorScheme.secondary,
                        fontSize   = 10.sp,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold
                    )
                }

                val (voiceLabel, voiceColor) = when (voiceState) {
                    VoskVoiceController.State.LISTENING       -> "🎙 SAY \"JOLT…\"" to MaterialTheme.colorScheme.secondary
                    VoskVoiceController.State.AWAKE_COMMAND   -> "🎙 LISTENING — SAY COMMAND…" to JoltColors.alertYellow
                    VoskVoiceController.State.AWAITING_REASON -> "🎙 REASON?" to JoltColors.alertYellow
                    VoskVoiceController.State.MUTED_BY_USER   -> "🔇 MIC MUTED" to Color.Gray
                    VoskVoiceController.State.INITIALIZING    -> "🎙 VOICE INIT…" to MaterialTheme.colorScheme.tertiary
                    VoskVoiceController.State.UNAVAILABLE     -> "🔇 VOICE OFF" to Color.Gray
                }
                Box(
                    modifier = Modifier
                        .background(Color.Black.copy(alpha = 0.72f), RoundedCornerShape(4.dp))
                        .clickable(
                            enabled = voiceState != VoskVoiceController.State.UNAVAILABLE &&
                                voiceState != VoskVoiceController.State.INITIALIZING,
                            onClick = onToggleVoiceMute
                        )
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text(
                        text       = voiceLabel,
                        color      = voiceColor,
                        fontSize   = 10.sp,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            if (telemetryRecording) {
                Box(
                    modifier = Modifier
                        .align(Alignment.TopEnd)
                        .padding(8.dp)
                        .background(Color.Black.copy(alpha = 0.72f), RoundedCornerShape(4.dp))
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text(
                        text       = "● REC TELEMETRY",
                        color      = MaterialTheme.colorScheme.primary,
                        fontSize   = 10.sp,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            if (activeOcr != null || activeMmc != null) {
                Box(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(bottom = 12.dp)
                        .background(Color.Black.copy(alpha = 0.82f), RoundedCornerShape(8.dp))
                        .padding(horizontal = 20.dp, vertical = 10.dp)
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        if (activeOcr != null) {
                            Text(
                                text       = activeOcr,
                                color      = Color.White,
                                fontSize   = 28.sp,
                                fontWeight = FontWeight.ExtraBold,
                                fontFamily = FontFamily.Monospace,
                                textAlign  = TextAlign.Center
                            )
                            Text(
                                text       = "LICENSE PLATE",
                                color      = MaterialTheme.colorScheme.secondary,
                                fontSize   = 9.sp,
                                fontFamily = FontFamily.Monospace,
                                textAlign  = TextAlign.Center
                            )
                        } else {
                            Text(
                                text       = activeMmc ?: "",
                                color      = MaterialTheme.colorScheme.tertiary,
                                fontSize   = 16.sp,
                                fontWeight = FontWeight.Bold,
                                textAlign  = TextAlign.Center
                            )
                            Text(
                                text       = "VEHICLE APPEARANCE",
                                color      = Color.Gray,
                                fontSize   = 9.sp,
                                fontFamily = FontFamily.Monospace,
                                textAlign  = TextAlign.Center
                            )
                        }
                    }
                }
            }
        }

        AnimatedVisibility(visible = recentFlagState != null) {
            val rf = recentFlagState
            if (rf != null) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(MaterialTheme.colorScheme.surface)
                        .horizontalScroll(rememberScrollState())
                        .padding(horizontal = 12.dp, vertical = 6.dp),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    OutlinedButton(
                        onClick = onUndoRecentFlag,
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                        modifier = Modifier.height(34.dp)
                    ) {
                        Text("UNDO", fontSize = 11.sp, fontWeight = FontWeight.Bold, color = JoltColors.alertYellow)
                    }
                    BehaviorTag.entries.forEach { tag ->
                        val selected = tag in rf.selectedTags
                        FilterChip(
                            selected = selected,
                            onClick  = { onToggleQuickTag(tag) },
                            label    = {
                                Text(
                                    text       = tag.shortHudCode,
                                    fontSize   = 11.sp,
                                    fontWeight = FontWeight.Bold,
                                    fontFamily = FontFamily.Monospace
                                )
                            }
                        )
                    }
                }
            }
        }

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .padding(horizontal = 14.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Button(
                onClick   = onFlagBadDriver,
                modifier  = Modifier.weight(2.8f).fillMaxHeight(),
                colors    = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                shape     = RoundedCornerShape(14.dp),
                elevation = ButtonDefaults.buttonElevation(defaultElevation = 8.dp)
            ) {
                Icon(
                    imageVector        = Icons.Default.Flag,
                    contentDescription = null,
                    tint               = Color.White,
                    modifier           = Modifier.size(26.dp)
                )
                Spacer(modifier = Modifier.width(10.dp))
                Text(
                    text          = "FLAG BAD",
                    color         = Color.White,
                    fontSize      = 19.sp,
                    fontWeight    = FontWeight.ExtraBold,
                    letterSpacing = 1.5.sp
                )
            }

            Button(
                onClick   = onFlagAngel,
                modifier  = Modifier.weight(1.2f).fillMaxHeight(),
                colors    = ButtonDefaults.buttonColors(containerColor = JoltColors.angelGreen),
                shape     = RoundedCornerShape(14.dp),
                elevation = ButtonDefaults.buttonElevation(defaultElevation = 6.dp)
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Icon(
                        imageVector        = Icons.Default.Star,
                        contentDescription = "Angel courteous vehicle",
                        tint               = Color.Black,
                        modifier           = Modifier.size(22.dp)
                    )
                    Text(
                        text       = "ANGEL",
                        color      = Color.Black,
                        fontSize   = 12.sp,
                        fontWeight = FontWeight.ExtraBold
                    )
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Map tab — OSMDroid map with severity-colored pins
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun JoltMapScreen(
    logs: List<DriverLog>,
    currentLocation: android.location.Location?
) {
    var mapViewRef    by remember { mutableStateOf<MapView?>(null) }
    var userHasPanned by remember { mutableStateOf(false) }
    val geoLogs = remember(logs) { logs.filter { it.latitude != 0.0 || it.longitude != 0.0 } }

    DisposableEffect(Unit) {
        onDispose { mapViewRef = null }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        JoltMapView(
            modifier        = Modifier.fillMaxSize(),
            logs            = geoLogs,
            currentLocation = currentLocation,
            userHasPanned   = userHasPanned,
            onUserPanned    = { userHasPanned = true },
            onMapReady      = { mapViewRef = it }
        )

        if (geoLogs.isEmpty()) {
            Box(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(top = 14.dp)
                    .background(Color.Black.copy(alpha = 0.70f), RoundedCornerShape(8.dp))
                    .padding(horizontal = 16.dp, vertical = 8.dp)
            ) {
                Text(
                    text      = "Pins appear here when you flag a driver on the Main tab",
                    color     = Color.Gray,
                    fontSize  = 11.sp,
                    textAlign = TextAlign.Center,
                    fontFamily = FontFamily.Monospace
                )
            }
        }

        if (geoLogs.isNotEmpty()) {
            Box(
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(12.dp)
                    .background(MaterialTheme.colorScheme.primary, RoundedCornerShape(4.dp))
                    .padding(horizontal = 10.dp, vertical = 4.dp)
            ) {
                Text(
                    text       = "${geoLogs.size} MAPPED",
                    color      = Color.White,
                    fontSize   = 10.sp,
                    fontWeight = FontWeight.ExtraBold,
                    fontFamily = FontFamily.Monospace
                )
            }
        }

        FloatingActionButton(
            onClick = {
                userHasPanned = false
                currentLocation?.let { loc ->
                    mapViewRef?.controller?.animateTo(GeoPoint(loc.latitude, loc.longitude))
                }
            },
            modifier       = Modifier.align(Alignment.BottomEnd).padding(16.dp),
            containerColor = JoltColors.actionBlue,
            contentColor   = Color.White
        ) {
            Icon(Icons.Default.MyLocation, contentDescription = "Re-center map")
        }
    }
}

@Composable
fun JoltMapView(
    modifier: Modifier,
    logs: List<DriverLog>,
    currentLocation: android.location.Location?,
    userHasPanned: Boolean,
    onUserPanned: () -> Unit,
    onMapReady: (MapView) -> Unit
) {
    val context        = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val dateFormat     = remember { SimpleDateFormat("MM/dd  HH:mm", Locale.getDefault()) }

    val makePin: (String) -> android.graphics.drawable.BitmapDrawable = { hexColor ->
        val dp  = context.resources.displayMetrics.density
        val sz  = (dp * 28).toInt()
        val bmp = Bitmap.createBitmap(sz, sz, Bitmap.Config.ARGB_8888)
        val cvs = android.graphics.Canvas(bmp)
        val paint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG)
        paint.color = android.graphics.Color.parseColor(hexColor)
        cvs.drawCircle(sz / 2f, sz / 2f, sz / 2f - dp, paint)
        paint.color = android.graphics.Color.WHITE
        paint.style = android.graphics.Paint.Style.STROKE
        paint.strokeWidth = dp * 2.5f
        cvs.drawCircle(sz / 2f, sz / 2f, sz / 2f - dp * 2.5f, paint)
        android.graphics.drawable.BitmapDrawable(context.resources, bmp)
    }
    val maroonPinDrawable = remember(context) { makePin("#AD1457") }
    val redPinDrawable    = remember(context) { makePin("#FF1744") }
    val orangePinDrawable = remember(context) { makePin("#FF9100") }
    val yellowPinDrawable = remember(context) { makePin("#FFD700") }
    val greenPinDrawable  = remember(context) { makePin("#00E676") }

    val mapViewHolder    = remember { mutableStateOf<MapView?>(null) }
    val lastRenderedLogs = remember { mutableStateOf<List<DriverLog>?>(null) }
    val lastCenteredLoc  = remember { mutableStateOf<android.location.Location?>(null) }

    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_RESUME -> mapViewHolder.value?.onResume()
                Lifecycle.Event.ON_PAUSE  -> mapViewHolder.value?.onPause()
                else -> {}
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose {
            lifecycleOwner.lifecycle.removeObserver(observer)
            mapViewHolder.value?.onDetach()
            mapViewHolder.value = null
        }
    }

    AndroidView(
        factory = { ctx ->
            MapView(ctx).also { mapView ->
                mapView.setTileSource(TileSourceFactory.MAPNIK)
                mapView.setUseDataConnection(true)
                mapView.setMultiTouchControls(true)
                mapView.controller.setZoom(15.0)
                mapView.controller.setCenter(GeoPoint(41.7658, -72.6734))
                mapView.setOnTouchListener { v, _ ->
                    onUserPanned()
                    v.performClick()
                    false
                }
                mapViewHolder.value = mapView
                onMapReady(mapView)
            }
        },
        update = { mapView ->
            if (lastRenderedLogs.value !== logs) {
                lastRenderedLogs.value = logs
                mapView.overlays.clear()
                logs.forEach { log ->
                    val tagsLabel = BehaviorTag.parseList(log.behaviorTags)
                        .joinToString(", ") { it.displayLabel }
                    val markerPin = when {
                        log.rating == "ANGEL"    -> greenPinDrawable
                        log.rating == "SIGHTING" -> yellowPinDrawable
                        else -> when (JoltSeverity.fromCode(log.severity)) {
                            JoltSeverity.MAROON -> maroonPinDrawable
                            JoltSeverity.RED    -> redPinDrawable
                            JoltSeverity.YELLOW -> yellowPinDrawable
                            else                -> orangePinDrawable
                        }
                    }
                    val marker = Marker(mapView).apply {
                        position = GeoPoint(log.latitude, log.longitude)
                        title    = log.plateOcr ?: log.vehicleMmc ?: "Flagged Vehicle"
                        snippet  = listOfNotNull(
                            log.severity ?: log.rating,
                            tagsLabel.ifEmpty { null },
                            dateFormat.format(Date(log.timestamp))
                        ).joinToString("  •  ")
                        icon     = markerPin
                        setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
                    }
                    mapView.overlays.add(marker)
                }
                mapView.invalidate()
            }

            if (!userHasPanned && currentLocation != null && currentLocation !== lastCenteredLoc.value) {
                lastCenteredLoc.value = currentLocation
                mapView.controller.animateTo(GeoPoint(currentLocation.latitude, currentLocation.longitude))
            }
        },
        modifier = modifier
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Phase 3 History & Trends Screen (Parked Review + Search/Filter/Edit + Analytics)
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun HistoryAndTrendsScreen(
    logList: List<DriverLog>,
    trendsReport: TrendsAnalyticsEngine.TrendsReport,
    onSaveEdits: (DriverLog, String, String, String, String, List<BehaviorTag>, JoltSeverity, String) -> Unit,
    onDeleteRecord: (DriverLog) -> Unit
) {
    var subTab by remember { mutableIntStateOf(0) } // 0 = Incidents, 1 = Trends
    var searchQuery by remember { mutableStateOf("") }
    var ratingFilter by remember { mutableStateOf("ALL") } // ALL, BAD, ANGEL, SIGHTING, PEER
    var editingLog by remember { mutableStateOf<DriverLog?>(null) }

    val activeEdit = editingLog
    if (activeEdit != null) {
        IncidentReviewEditDialog(
            log      = activeEdit,
            onDismiss = { editingLog = null },
            onSave   = { plate, color, make, body, tags, sev, notes ->
                onSaveEdits(activeEdit, plate, color, make, body, tags, sev, notes)
                editingLog = null
            },
            onDelete = {
                onDeleteRecord(activeEdit)
                editingLog = null
            }
        )
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(horizontal = 16.dp, vertical = 12.dp)
    ) {
        // Segmented toggle: [ 📋 Incidents ] [ 📊 Trends ]
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(MaterialTheme.colorScheme.surface, RoundedCornerShape(10.dp))
                .padding(4.dp),
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            val tabItems = listOf("INCIDENTS (${logList.size})" to 0, "TRENDS & RATES" to 1)
            tabItems.forEach { (title, idx) ->
                val selected = subTab == idx
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .clip(RoundedCornerShape(8.dp))
                        .background(if (selected) MaterialTheme.colorScheme.primary else Color.Transparent)
                        .clickable { subTab = idx }
                        .padding(vertical = 9.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text       = title,
                        color      = if (selected) Color.White else Color.Gray,
                        fontWeight = FontWeight.ExtraBold,
                        fontSize   = 11.sp,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(10.dp))

        if (subTab == 0) {
            // Search box
            OutlinedTextField(
                value         = searchQuery,
                onValueChange = { searchQuery = it },
                placeholder   = { Text("Search plate, make, color, body, tag…", fontSize = 12.sp) },
                leadingIcon   = { Icon(Icons.Default.Search, contentDescription = null, modifier = Modifier.size(18.dp)) },
                trailingIcon  = {
                    if (searchQuery.isNotEmpty()) {
                        IconButton(onClick = { searchQuery = "" }, modifier = Modifier.size(24.dp)) {
                            Icon(Icons.Default.Clear, contentDescription = "Clear")
                        }
                    }
                },
                singleLine = true,
                modifier   = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(6.dp))

            // Quick filter chips
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                listOf("ALL", "BAD", "ANGEL", "SIGHTING", "PEER").forEach { key ->
                    FilterChip(
                        selected = ratingFilter == key,
                        onClick  = { ratingFilter = key },
                        label    = {
                            Text(
                                text       = if (key == "ANGEL") "★ ANGEL" else key,
                                fontSize   = 10.sp,
                                fontWeight = FontWeight.Bold,
                                fontFamily = FontFamily.Monospace
                            )
                        }
                    )
                }
            }

            Spacer(modifier = Modifier.height(6.dp))

            val filteredLogs = remember(logList, searchQuery, ratingFilter) {
                val q = searchQuery.trim().lowercase()
                logList.filter { row ->
                    val matchesFilter = when (ratingFilter) {
                        "BAD"      -> row.rating == "BAD"
                        "ANGEL"    -> row.rating == "ANGEL"
                        "SIGHTING" -> row.rating == "SIGHTING"
                        "PEER"     -> row.reporterId != "LOCAL"
                        else       -> true
                    }
                    if (!matchesFilter) return@filter false
                    if (q.isEmpty()) return@filter true
                    val haystack = listOfNotNull(
                        row.plateOcr, row.plateNormalized, row.vehicleMmc,
                        row.vehicleColor, row.vehicleMake, row.vehicleBodyStyle,
                        row.behaviorTags, row.severity, row.notes, row.reporterId
                    ).joinToString(" ").lowercase()
                    haystack.contains(q)
                }
            }

            if (filteredLogs.isEmpty()) {
                EmptyState(
                    icon = Icons.Default.List,
                    text = if (logList.isEmpty()) {
                        "No captures yet.\nSay \"Jolt, flag…\" on the Main tab."
                    } else {
                        "No records match your search/filter."
                    }
                )
            } else {
                Text(
                    text       = "Tap any BAD / ANGEL card while parked to review or edit",
                    color      = Color.Gray,
                    fontSize   = 10.sp,
                    fontFamily = FontFamily.Monospace,
                    modifier   = Modifier.padding(bottom = 6.dp)
                )
                LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(filteredLogs, key = { it.id }) { log ->
                        DriverLogItemWidget(
                            log     = log,
                            onClick = if (log.rating != "SIGHTING") ({ editingLog = log }) else null
                        )
                    }
                }
            }
        } else {
            TrendsDashboardContent(report = trendsReport)
        }
    }
}

@Composable
private fun TrendsDashboardContent(report: TrendsAnalyticsEngine.TrendsReport) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        // 1. Overview Counters Row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            MetricMiniCard("BAD FLAGS", report.totalBadIncidents.toString(), MaterialTheme.colorScheme.primary, Modifier.weight(1f))
            MetricMiniCard("PLATES", report.uniqueFlaggedPlates.toString(), Color.White, Modifier.weight(1f))
            MetricMiniCard("★ ANGELS", report.totalAngels.toString(), JoltColors.angelGreen, Modifier.weight(1f))
            MetricMiniCard("SIGHTINGS", report.totalSightings.toString(), JoltColors.alertYellow, Modifier.weight(1f))
        }

        // 2. White SUV Hypothesis Spotlight Card
        val ws = report.whiteSuvSpotlight
        Card(
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            shape  = RoundedCornerShape(10.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(14.dp)) {
                Text(
                    text       = "HYPOTHESIS TRACKER • WHITE SUVs",
                    color      = MaterialTheme.colorScheme.tertiary,
                    fontWeight = FontWeight.ExtraBold,
                    fontSize   = 11.sp,
                    fontFamily = FontFamily.Monospace
                )
                Spacer(modifier = Modifier.height(8.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Column(
                        modifier = Modifier
                            .weight(1f)
                            .background(JoltColors.cardElevated, RoundedCornerShape(8.dp))
                            .padding(10.dp)
                    ) {
                        Text("WHITE SUV", color = Color.Gray, fontSize = 10.sp, fontFamily = FontFamily.Monospace)
                        Text(
                            text       = "${"%.1f".format(ws.whiteSuvRatePct)}%",
                            color      = Color.White,
                            fontWeight = FontWeight.ExtraBold,
                            fontSize   = 20.sp,
                            fontFamily = FontFamily.Monospace
                        )
                        Text(
                            text     = "${ws.flaggedWhiteSuvs} flagged / ${ws.observedWhiteSuvs} seen",
                            color    = Color.Gray,
                            fontSize = 10.sp
                        )
                    }
                    Column(
                        modifier = Modifier
                            .weight(1f)
                            .background(JoltColors.cardElevated, RoundedCornerShape(8.dp))
                            .padding(10.dp)
                    ) {
                        Text("ALL OTHER TRAFFIC", color = Color.Gray, fontSize = 10.sp, fontFamily = FontFamily.Monospace)
                        Text(
                            text       = "${"%.1f".format(ws.allOtherRatePct)}%",
                            color      = Color.White,
                            fontWeight = FontWeight.ExtraBold,
                            fontSize   = 20.sp,
                            fontFamily = FontFamily.Monospace
                        )
                        Text(
                            text     = "${ws.allOtherFlagged} flagged / ${ws.allOtherObserved} seen",
                            color    = Color.Gray,
                            fontSize = 10.sp
                        )
                    }
                }
                if (ws.relativeMultiplier > 0f) {
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text       = "Relative Rate: ${"%.2f".format(ws.relativeMultiplier)}× baseline encounter rate",
                        color      = if (ws.relativeMultiplier >= 1.15f) JoltColors.severityOrange else JoltColors.angelGreen,
                        fontSize   = 11.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }
        }

        // 3. Central CT Make Watchlist (Honda / Acura / Infiniti)
        Card(
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            shape  = RoundedCornerShape(10.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(14.dp)) {
                Text(
                    text       = "CENTRAL CT WATCHLIST (HONDA • ACURA • INFINITI)",
                    color      = JoltColors.watchlistAccent,
                    fontWeight = FontWeight.ExtraBold,
                    fontSize   = 11.sp,
                    fontFamily = FontFamily.Monospace
                )
                Spacer(modifier = Modifier.height(8.dp))
                report.watchlistRows.forEach { row ->
                    RateBarRow(row = row, accentColor = JoltColors.watchlistAccent)
                    Spacer(modifier = Modifier.height(6.dp))
                }
            }
        }

        // 4. All Observed & Flagged Makes Leaderboard
        if (report.makeLeaderboard.isNotEmpty()) {
            AnalyticsSectionCard(
                title = "INCIDENTS & RATE BY MAKE",
                subtitle = "Shows both raw flags and normalized (flags ÷ encounters)",
                rows = report.makeLeaderboard.take(10),
                barColor = MaterialTheme.colorScheme.primary
            )
        }

        // 5. Body Style & Paint Color Breakdown
        if (report.bodyLeaderboard.isNotEmpty()) {
            AnalyticsSectionCard(
                title = "BY BODY STYLE",
                subtitle = "SUV • Sedan • Pickup • Van • Coupe • Wagon/Hatch",
                rows = report.bodyLeaderboard,
                barColor = MaterialTheme.colorScheme.tertiary
            )
        }

        if (report.colorLeaderboard.isNotEmpty()) {
            AnalyticsSectionCard(
                title = "BY VEHICLE COLOR",
                subtitle = "Extracted from vehicle crop HSV + voice descriptors",
                rows = report.colorLeaderboard,
                barColor = JoltColors.severityOrange
            )
        }

        // 6. Top Behavior Triggers
        if (report.triggerBreakdown.isNotEmpty()) {
            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                shape  = RoundedCornerShape(10.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(14.dp)) {
                    Text(
                        text       = "MOST COMMON BEHAVIOR TRIGGERS",
                        color      = Color.White,
                        fontWeight = FontWeight.ExtraBold,
                        fontSize   = 11.sp,
                        fontFamily = FontFamily.Monospace
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    report.triggerBreakdown.forEach { trig ->
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(vertical = 3.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text       = trig.tag.shortHudCode,
                                color      = severityUiColor(trig.tag.defaultSeverity),
                                fontWeight = FontWeight.Bold,
                                fontSize   = 11.sp,
                                fontFamily = FontFamily.Monospace
                            )
                            Text(
                                text       = "${trig.count} (${"%.0f".format(trig.sharePct)}%)",
                                color      = Color.Gray,
                                fontSize   = 11.sp,
                                fontFamily = FontFamily.Monospace
                            )
                        }
                    }
                }
            }
        }

        // 7. Repeat Offenders Leaderboard
        if (report.repeatOffenders.isNotEmpty()) {
            Card(
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
                shape  = RoundedCornerShape(10.dp),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(14.dp)) {
                    Text(
                        text       = "REPEAT ENCOUNTERS & FREQUENT OFFENDERS",
                        color      = JoltColors.severityRed,
                        fontWeight = FontWeight.ExtraBold,
                        fontSize   = 11.sp,
                        fontFamily = FontFamily.Monospace
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    report.repeatOffenders.forEach { rep ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 4.dp)
                                .background(JoltColors.cardElevated, RoundedCornerShape(6.dp))
                                .padding(horizontal = 10.dp, vertical = 6.dp),
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text       = "${rep.plateDisplay}  ${"●".repeat(rep.badIncidentCount.coerceAtMost(5))}",
                                    color      = severityUiColor(rep.effectiveSeverity),
                                    fontWeight = FontWeight.ExtraBold,
                                    fontSize   = 12.sp,
                                    fontFamily = FontFamily.Monospace
                                )
                                Text(
                                    text  = listOfNotNull(rep.primaryHudBehavior, rep.vehicleMmc).joinToString(" • "),
                                    color = Color.Gray,
                                    fontSize = 10.sp
                                )
                            }
                            Text(
                                text       = "${rep.badIncidentCount} flag / ${rep.sightingCount} seen",
                                color      = Color.White,
                                fontSize   = 10.sp,
                                fontFamily = FontFamily.Monospace
                            )
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))
    }
}

@Composable
private fun MetricMiniCard(label: String, value: String, valueColor: Color, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .background(MaterialTheme.colorScheme.surface, RoundedCornerShape(8.dp))
            .padding(vertical = 10.dp, horizontal = 6.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text       = value,
            color      = valueColor,
            fontWeight = FontWeight.ExtraBold,
            fontSize   = 18.sp,
            fontFamily = FontFamily.Monospace
        )
        Text(
            text       = label,
            color      = Color.Gray,
            fontSize   = 9.sp,
            fontFamily = FontFamily.Monospace
        )
    }
}

@Composable
private fun AnalyticsSectionCard(
    title: String,
    subtitle: String,
    rows: List<TrendsAnalyticsEngine.CategoryRateRow>,
    barColor: Color
) {
    Card(
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        shape  = RoundedCornerShape(10.dp),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(
                text       = title,
                color      = Color.White,
                fontWeight = FontWeight.ExtraBold,
                fontSize   = 11.sp,
                fontFamily = FontFamily.Monospace
            )
            Text(text = subtitle, color = Color.Gray, fontSize = 10.sp)
            Spacer(modifier = Modifier.height(8.dp))
            rows.forEach { row ->
                RateBarRow(row = row, accentColor = if (row.isWatchlist) JoltColors.watchlistAccent else barColor)
                Spacer(modifier = Modifier.height(6.dp))
            }
        }
    }
}

@Composable
private fun RateBarRow(row: TrendsAnalyticsEngine.CategoryRateRow, accentColor: Color) {
    Column(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text       = row.label,
                color      = Color.White,
                fontWeight = FontWeight.SemiBold,
                fontSize   = 12.sp
            )
            Text(
                text       = "${row.flaggedCount} flag / ${row.observedEncounters} seen (${"%.0f".format(row.ratePct)}%)",
                color      = Color.Gray,
                fontSize   = 11.sp,
                fontFamily = FontFamily.Monospace
            )
        }
        Spacer(modifier = Modifier.height(3.dp))
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(6.dp)
                .clip(RoundedCornerShape(3.dp))
                .background(JoltColors.barTrack)
        ) {
            val fillFrac = (row.ratePct / 100f).coerceIn(0f, 1f)
            if (fillFrac > 0f) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth(fillFrac)
                        .fillMaxHeight()
                        .background(accentColor)
                )
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parked Incident Review & Correction Dialog (Requirements §13 Correctability)
// ─────────────────────────────────────────────────────────────────────────────

@Composable
private fun IncidentReviewEditDialog(
    log: DriverLog,
    onDismiss: () -> Unit,
    onSave: (String, String, String, String, List<BehaviorTag>, JoltSeverity, String) -> Unit,
    onDelete: () -> Unit
) {
    val resolvedApp = remember(log) { PartialMatchEngine.resolveStoredAppearance(listOf(log)) }
    var plateText by remember(log) { mutableStateOf(log.plateOcr ?: "") }
    var colorText by remember(log) { mutableStateOf(log.vehicleColor ?: resolvedApp.color ?: "") }
    var makeText  by remember(log) { mutableStateOf(log.vehicleMake ?: resolvedApp.make ?: "") }
    var bodyText  by remember(log) { mutableStateOf(log.vehicleBodyStyle ?: resolvedApp.bodyStyle ?: "") }
    var notesText by remember(log) { mutableStateOf(log.notes ?: "") }
    var selectedTags by remember(log) { mutableStateOf(BehaviorTag.parseList(log.behaviorTags)) }
    var selectedSeverity by remember(log) {
        mutableStateOf(JoltSeverity.fromCode(log.severity) ?: JoltSeverity.ORANGE)
    }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = {
            Text(
                text       = if (log.rating == "ANGEL") "Review ★ Angel Record" else "Review / Edit Incident",
                fontWeight = FontWeight.ExtraBold,
                fontSize   = 16.sp
            )
        },
        text = {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                OutlinedTextField(
                    value = plateText,
                    onValueChange = { plateText = it.uppercase() },
                    label = { Text("License Plate") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth()
                )
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    OutlinedTextField(
                        value = colorText,
                        onValueChange = { colorText = it },
                        label = { Text("Color") },
                        singleLine = true,
                        modifier = Modifier.weight(1f)
                    )
                    OutlinedTextField(
                        value = makeText,
                        onValueChange = { makeText = it },
                        label = { Text("Make") },
                        singleLine = true,
                        modifier = Modifier.weight(1f)
                    )
                }
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    OutlinedTextField(
                        value = bodyText,
                        onValueChange = { bodyText = it },
                        label = { Text("Body (SUV/Sedan/Pickup)") },
                        singleLine = true,
                        modifier = Modifier.weight(1f)
                    )
                }

                if (log.rating != "ANGEL") {
                    Text("Severity:", fontSize = 11.sp, color = Color.Gray, fontFamily = FontFamily.Monospace)
                    Row(
                        modifier = Modifier.horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        listOf(JoltSeverity.YELLOW, JoltSeverity.ORANGE, JoltSeverity.RED, JoltSeverity.MAROON).forEach { sev ->
                            FilterChip(
                                selected = selectedSeverity == sev,
                                onClick  = { selectedSeverity = sev },
                                label    = { Text(sev.displayLabel, fontSize = 10.sp, fontWeight = FontWeight.Bold) }
                            )
                        }
                    }

                    Text("Behavior Tags (up to 2):", fontSize = 11.sp, color = Color.Gray, fontFamily = FontFamily.Monospace)
                    Row(
                        modifier = Modifier.horizontalScroll(rememberScrollState()),
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        BehaviorTag.entries.forEach { tag ->
                            val isSel = tag in selectedTags
                            FilterChip(
                                selected = isSel,
                                onClick = {
                                    selectedTags = if (isSel) {
                                        selectedTags - tag
                                    } else {
                                        (selectedTags + tag).takeLast(BehaviorTag.MAX_TAGS_PER_INCIDENT)
                                    }
                                },
                                label = { Text(tag.shortHudCode, fontSize = 10.sp) }
                            )
                        }
                    }
                }

                OutlinedTextField(
                    value = notesText,
                    onValueChange = { notesText = it },
                    label = { Text("Distinctive Features / Notes (tint, stickers, racks)") },
                    modifier = Modifier.fillMaxWidth()
                )

                TextButton(onClick = onDelete) {
                    Icon(Icons.Default.Delete, contentDescription = null, tint = MaterialTheme.colorScheme.primary)
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Delete This Record", color = MaterialTheme.colorScheme.primary)
                }
            }
        },
        confirmButton = {
            Button(
                onClick = {
                    onSave(plateText, colorText, makeText, bodyText, selectedTags, selectedSeverity, notesText)
                }
            ) {
                Text("Save")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Cancel") }
        }
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Export tab (CSV + Full DB Backup + Trusted-Group Sync + Training + Telemetry)
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun ExportScreen(
    logList: List<DriverLog>,
    trustedUnitId: String,
    onExportGroupBundle: () -> Unit,
    onImportGroupBundle: (Uri) -> Unit,
    trainingMode: Boolean,
    trainingCount: Int,
    onToggleTraining: () -> Unit,
    onExportTraining: () -> Unit,
    onClearTraining: () -> Unit,
    telemetryEnabled: Boolean,
    telemetryCount: Int,
    onToggleTelemetry: () -> Unit,
    onExportTelemetry: () -> Unit,
    onClearTelemetry: () -> Unit,
    onBackupDatabase: () -> Unit
) {
    val context = LocalContext.current
    val importBundleLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        if (uri != null) onImportGroupBundle(uri)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text       = "EXPORT & GROUP SYNC",
            fontWeight = FontWeight.Bold,
            fontSize   = 16.sp,
            color      = Color.Gray,
            fontFamily = FontFamily.Monospace,
            modifier   = Modifier.padding(bottom = 20.dp)
        )

        Button(
            onClick  = { exportToCsv(context, logList) },
            modifier = Modifier.fillMaxWidth().height(52.dp),
            colors   = ButtonDefaults.buttonColors(containerColor = JoltColors.actionBlue),
            shape    = RoundedCornerShape(10.dp)
        ) {
            Icon(Icons.Default.FileDownload, contentDescription = null)
            Spacer(modifier = Modifier.width(10.dp))
            Text("Export Records (CSV)", fontWeight = FontWeight.Bold, fontSize = 14.sp)
        }

        Spacer(modifier = Modifier.height(8.dp))
        Text(
            text     = "${logList.size} record${if (logList.size != 1) "s" else ""} in database",
            color    = Color.Gray,
            fontSize = 12.sp
        )

        Spacer(modifier = Modifier.height(Dimens.spaceMd))

        Button(
            onClick  = onBackupDatabase,
            enabled  = logList.isNotEmpty(),
            modifier = Modifier.fillMaxWidth().height(52.dp),
            colors   = ButtonDefaults.buttonColors(containerColor = JoltColors.actionBlue),
            shape    = RoundedCornerShape(10.dp)
        ) {
            Icon(Icons.Default.Save, contentDescription = null)
            Spacer(modifier = Modifier.width(10.dp))
            Text("Backup Database + Crops (ZIP)", fontWeight = FontWeight.Bold, fontSize = 14.sp)
        }

        Spacer(modifier = Modifier.height(Dimens.spaceLg))

        // Phase 3 Closed Trusted-Group Share / Import Card (3-4 family/commuter units)
        Card(
            colors   = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            shape    = RoundedCornerShape(10.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(Dimens.spaceLg)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text       = "Trusted-Group Share",
                        fontWeight = FontWeight.SemiBold,
                        color      = Color.White,
                        fontSize   = 14.sp
                    )
                    Text(
                        text       = trustedUnitId,
                        color      = JoltColors.peerBadge,
                        fontWeight = FontWeight.Bold,
                        fontSize   = 11.sp,
                        fontFamily = FontFamily.Monospace
                    )
                }
                Spacer(modifier = Modifier.height(Dimens.spaceXs))
                Text(
                    text     = "Share vehicle warnings with 3–4 family or coworker phones. " +
                               "GPS coordinates are stripped automatically for privacy.",
                    color    = Color.Gray,
                    fontSize = 11.sp
                )
                Spacer(modifier = Modifier.height(Dimens.spaceMd))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(Dimens.spaceSm)
                ) {
                    Button(
                        onClick  = onExportGroupBundle,
                        enabled  = logList.isNotEmpty(),
                        modifier = Modifier.weight(1f).height(46.dp),
                        colors   = ButtonDefaults.buttonColors(containerColor = JoltColors.actionBlue),
                        shape    = RoundedCornerShape(10.dp)
                    ) {
                        Text("Share (.json)", fontWeight = FontWeight.Bold, fontSize = 12.sp)
                    }
                    OutlinedButton(
                        onClick  = { importBundleLauncher.launch(arrayOf("application/json", "*/*")) },
                        modifier = Modifier.weight(1f).height(46.dp),
                        shape    = RoundedCornerShape(10.dp)
                    ) {
                        Text("Import Peer", fontWeight = FontWeight.Bold, fontSize = 12.sp)
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(Dimens.spaceLg))

        // Training data collection + export card
        Card(
            colors   = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            shape    = RoundedCornerShape(10.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(Dimens.spaceLg)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text       = "Training Data Collection",
                        fontWeight = FontWeight.SemiBold,
                        color      = Color.White,
                        fontSize   = 14.sp
                    )
                    Switch(checked = trainingMode, onCheckedChange = { onToggleTraining() })
                }
                Spacer(modifier = Modifier.height(Dimens.spaceXs))
                Text(
                    text     = "Saves annotated YOLO frames asynchronously on background I/O (keep OFF for normal commuting).",
                    color    = Color.Gray,
                    fontSize = 11.sp
                )
                Spacer(modifier = Modifier.height(Dimens.spaceMd))
                Text(
                    text       = "$trainingCount sample${if (trainingCount != 1) "s" else ""} collected",
                    color      = Color.Gray,
                    fontSize   = 12.sp,
                    fontFamily = FontFamily.Monospace
                )
                Spacer(modifier = Modifier.height(Dimens.spaceSm))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(Dimens.spaceSm)
                ) {
                    Button(
                        onClick  = onExportTraining,
                        enabled  = trainingCount > 0,
                        modifier = Modifier.weight(1f).height(48.dp),
                        colors   = ButtonDefaults.buttonColors(containerColor = JoltColors.actionBlue),
                        shape    = RoundedCornerShape(10.dp)
                    ) {
                        Icon(Icons.Default.FileDownload, contentDescription = null)
                        Spacer(modifier = Modifier.width(Dimens.spaceSm))
                        Text("Export ZIP", fontWeight = FontWeight.Bold, fontSize = 13.sp)
                    }
                    OutlinedButton(
                        onClick  = onClearTraining,
                        enabled  = trainingCount > 0,
                        modifier = Modifier.height(48.dp),
                        shape    = RoundedCornerShape(10.dp)
                    ) {
                        Icon(Icons.Default.Delete, contentDescription = "Clear training samples")
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(Dimens.spaceLg))

        // Field Telemetry & Soak Logger card
        Card(
            colors   = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            shape    = RoundedCornerShape(10.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(Dimens.spaceLg)) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        text       = "Field Telemetry Logger",
                        fontWeight = FontWeight.SemiBold,
                        color      = Color.White,
                        fontSize   = 14.sp
                    )
                    Switch(checked = telemetryEnabled, onCheckedChange = { onToggleTelemetry() })
                }
                Spacer(modifier = Modifier.height(Dimens.spaceXs))
                Text(
                    text     = "Logs frame latency, thermal throttling/headroom, battery °C, " +
                               "Java/native heap, GPS speed, OCR votes & MMC scores to disk.",
                    color    = Color.Gray,
                    fontSize = 11.sp
                )
                Spacer(modifier = Modifier.height(Dimens.spaceMd))
                Text(
                    text       = "$telemetryCount event${if (telemetryCount != 1) "s" else ""} logged",
                    color      = if (telemetryEnabled) MaterialTheme.colorScheme.secondary else Color.Gray,
                    fontSize   = 12.sp,
                    fontFamily = FontFamily.Monospace
                )
                Spacer(modifier = Modifier.height(Dimens.spaceSm))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(Dimens.spaceSm)
                ) {
                    Button(
                        onClick  = onExportTelemetry,
                        enabled  = telemetryCount > 0,
                        modifier = Modifier.weight(1f).height(48.dp),
                        colors   = ButtonDefaults.buttonColors(containerColor = JoltColors.actionBlue),
                        shape    = RoundedCornerShape(10.dp)
                    ) {
                        Icon(Icons.Default.FileDownload, contentDescription = null)
                        Spacer(modifier = Modifier.width(Dimens.spaceSm))
                        Text("Export Report (.txt)", fontWeight = FontWeight.Bold, fontSize = 13.sp)
                    }
                    OutlinedButton(
                        onClick  = onClearTelemetry,
                        enabled  = telemetryCount > 0,
                        modifier = Modifier.height(48.dp),
                        shape    = RoundedCornerShape(10.dp)
                    ) {
                        Icon(Icons.Default.Delete, contentDescription = "Clear telemetry logs")
                    }
                }
            }
        }
    }
}

private object PlateThumbnailCache {
    private val cache = object : LruCache<String, Bitmap>(4 * 1024 * 1024) {
        override fun sizeOf(key: String, value: Bitmap): Int = value.allocationByteCount
    }

    fun get(path: String): Bitmap? = cache.get(path)?.takeIf { !it.isRecycled }
    fun put(path: String, bmp: Bitmap) {
        if (!bmp.isRecycled) cache.put(path, bmp)
    }
}

private fun decodeSampledThumbnail(path: String, reqWidth: Int = 216, reqHeight: Int = 108): Bitmap? {
    return try {
        val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(path, opts)
        var inSampleSize = 1
        val (h, w) = opts.outHeight to opts.outWidth
        if (h > reqHeight || w > reqWidth) {
            val halfH = h / 2
            val halfW = w / 2
            while (halfH / inSampleSize >= reqHeight && halfW / inSampleSize >= reqWidth) {
                inSampleSize *= 2
            }
        }
        BitmapFactory.Options().apply { this.inSampleSize = inSampleSize }.let {
            BitmapFactory.decodeFile(path, it)
        }
    } catch (_: Exception) {
        null
    }
}

internal fun csvField(value: String?): String {
    val v = value ?: return ""
    return "\"" + v.replace("\"", "\"\"") + "\""
}

private fun exportToCsv(context: Context, logs: List<DriverLog>) {
    try {
        val ts       = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val fileName = "jolt_export_$ts.csv"
        val values   = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, fileName)
            put(MediaStore.Downloads.MIME_TYPE, "text/csv")
        }
        val uri = context.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
        uri?.let {
            context.contentResolver.openOutputStream(it)?.use { stream ->
                stream.write(
                    ("ID,Timestamp,Latitude,Longitude,PlateOCR,PlateNormalized,VehicleMMC," +
                        "VehicleColor,VehicleMake,VehicleBodyStyle,ReporterId," +
                        "Rating,Severity,BehaviorTags,MatchConfidence,Notes,Battery,PlateCropPath\n").toByteArray()
                )
                logs.forEach { log ->
                    val line = listOf(
                        log.id.toString(),
                        log.timestamp.toString(),
                        log.latitude.toString(),
                        log.longitude.toString(),
                        csvField(log.plateOcr),
                        csvField(log.plateNormalized),
                        csvField(log.vehicleMmc),
                        csvField(log.vehicleColor),
                        csvField(log.vehicleMake),
                        csvField(log.vehicleBodyStyle),
                        csvField(log.reporterId),
                        log.rating,
                        csvField(log.severity),
                        csvField(log.behaviorTags),
                        log.matchConfidence.toString(),
                        csvField(log.notes),
                        log.batteryLevel.toString(),
                        csvField(log.plateCropPath)
                    ).joinToString(",") + "\n"
                    stream.write(line.toByteArray())
                }
            }
            Toast.makeText(context, "Exported to Downloads/$fileName", Toast.LENGTH_LONG).show()
        }
    } catch (e: Exception) {
        Log.e("ExportScreen", "CSV export failed: ${e.localizedMessage}", e)
        Toast.makeText(context, "Export failed: ${e.localizedMessage}", Toast.LENGTH_LONG).show()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Manage tab
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun ManageScreen(
    logList: List<DriverLog>,
    onDeleteLog: (DriverLog) -> Unit,
    onClearAll: () -> Unit
) {
    var showClearDialog by remember { mutableStateOf(false) }

    if (showClearDialog) {
        AlertDialog(
            onDismissRequest = { showClearDialog = false },
            title            = { Text("Clear All Records?") },
            text             = { Text("This will permanently delete all ${logList.size} captures and encounter stats.") },
            confirmButton    = {
                TextButton(onClick = { onClearAll(); showClearDialog = false }) {
                    Text("Delete All", color = MaterialTheme.colorScheme.primary)
                }
            },
            dismissButton    = {
                TextButton(onClick = { showClearDialog = false }) { Text("Cancel") }
            }
        )
    }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(bottom = 12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text       = "MANAGE RECORDS",
                fontWeight = FontWeight.Bold,
                fontSize   = 13.sp,
                color      = Color.Gray,
                fontFamily = FontFamily.Monospace
            )
            if (logList.isNotEmpty()) {
                Button(
                    onClick = { showClearDialog = true },
                    colors  = ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.primary),
                    contentPadding = PaddingValues(horizontal = 14.dp, vertical = 6.dp)
                ) {
                    Text("Clear All", color = Color.White, fontSize = 12.sp)
                }
            }
        }

        if (logList.isEmpty()) {
            EmptyState(
                icon = Icons.Default.ManageSearch,
                text = "No records to manage."
            )
        } else {
            LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(logList, key = { it.id }) { log ->
                    DriverLogManageItem(log = log, onDelete = { onDeleteLog(log) })
                }
            }
        }
    }
}

@Composable
private fun DriverLogManageItem(log: DriverLog, onDelete: () -> Unit) {
    val fmt = remember(log.timestamp) {
        SimpleDateFormat("MM/dd/yyyy HH:mm:ss", Locale.getDefault()).format(Date(log.timestamp))
    }
    Card(
        shape    = RoundedCornerShape(8.dp),
        colors   = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier          = Modifier.padding(12.dp).fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text       = log.plateOcr ?: log.vehicleMmc ?: log.notes ?: "Unidentified",
                    color      = Color.White,
                    fontWeight = FontWeight.Bold,
                    fontSize   = 13.sp,
                    fontFamily = if (log.plateOcr != null) FontFamily.Monospace else FontFamily.Default
                )
                Text(
                    text     = if (log.latitude != 0.0 || log.longitude != 0.0) {
                        "GPS: [${String.format("%.4f", log.latitude)}, ${String.format("%.4f", log.longitude)}]"
                    } else {
                        "Source: ${log.reporterId}"
                    },
                    color    = Color.Gray,
                    fontSize = 9.sp,
                    fontFamily = FontFamily.Monospace
                )
                Text(text = fmt, color = Color.Gray, fontSize = 9.sp)
            }
            IconButton(onClick = onDelete) {
                Icon(Icons.Default.Delete, contentDescription = "Delete", tint = MaterialTheme.colorScheme.primary)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared UI: empty-state hint + history log item card
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun EmptyState(icon: ImageVector, text: String, modifier: Modifier = Modifier) {
    Box(modifier = modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(
                imageVector        = icon,
                contentDescription = null,
                tint               = Color.Gray,
                modifier           = Modifier.size(Dimens.emptyIcon)
            )
            Spacer(modifier = Modifier.height(Dimens.spaceMd))
            Text(
                text       = text,
                color      = Color.Gray,
                fontSize   = 13.sp,
                textAlign  = TextAlign.Center,
                fontFamily = FontFamily.Monospace
            )
        }
    }
}

@Composable
fun DriverLogItemWidget(log: DriverLog, onClick: (() -> Unit)? = null) {
    val parsedSeverity = JoltSeverity.fromCode(log.severity)
    val ratingColor = when {
        log.rating == "ANGEL"    -> JoltColors.angelGreen
        log.rating == "SIGHTING" -> JoltColors.alertYellow
        parsedSeverity != null   -> severityUiColor(parsedSeverity)
        else                     -> MaterialTheme.colorScheme.primary
    }
    val badgeText = when (log.rating) {
        "ANGEL"    -> "★ ANGEL"
        "SIGHTING" -> "SEEN"
        else       -> parsedSeverity?.displayLabel ?: log.rating
    }
    val tagsSummary = remember(log.behaviorTags) {
        BehaviorTag.parseList(log.behaviorTags).joinToString(" • ") { it.shortHudCode }
    }
    val formattedDate = remember(log.timestamp) {
        SimpleDateFormat("MM/dd HH:mm", Locale.getDefault()).format(Date(log.timestamp))
    }
    val cropBitmap by produceState<Bitmap?>(initialValue = log.plateCropPath?.let { PlateThumbnailCache.get(it) }, log.plateCropPath) {
        val path = log.plateCropPath
        if (path == null) {
            value = null
            return@produceState
        }
        val cached = PlateThumbnailCache.get(path)
        if (cached != null) {
            value = cached
            return@produceState
        }
        value = withContext(Dispatchers.IO) {
            decodeSampledThumbnail(path)?.also { PlateThumbnailCache.put(path, it) }
        }
    }
    Card(
        shape    = RoundedCornerShape(8.dp),
        colors   = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        modifier = Modifier
            .fillMaxWidth()
            .then(if (onClick != null) Modifier.clickable(onClick = onClick) else Modifier)
    ) {
        Row(
            modifier              = Modifier.padding(12.dp).fillMaxWidth(),
            verticalAlignment     = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column(modifier = Modifier.width(60.dp)) {
                Text(
                    text       = badgeText,
                    color      = ratingColor,
                    fontWeight = FontWeight.Bold,
                    fontSize   = 10.sp,
                    fontFamily = FontFamily.Monospace
                )
                if (log.reporterId != "LOCAL") {
                    Text(
                        text       = "PEER",
                        color      = JoltColors.peerBadge,
                        fontWeight = FontWeight.Bold,
                        fontSize   = 9.sp,
                        fontFamily = FontFamily.Monospace
                    )
                }
            }
            Column(modifier = Modifier.weight(1f).padding(horizontal = 8.dp)) {
                if (log.plateOcr != null) {
                    Text(
                        text       = "Plate: ${log.plateOcr}",
                        color      = Color.White,
                        fontWeight = FontWeight.Bold,
                        fontSize   = 13.sp,
                        fontFamily = FontFamily.Monospace
                    )
                } else {
                    Text(
                        text       = log.vehicleMmc ?: log.notes ?: "Unidentified",
                        color      = MaterialTheme.colorScheme.tertiary,
                        fontWeight = FontWeight.SemiBold,
                        fontSize   = 12.sp
                    )
                }
                if (tagsSummary.isNotEmpty()) {
                    Text(
                        text       = tagsSummary,
                        color      = ratingColor,
                        fontWeight = FontWeight.SemiBold,
                        fontSize   = 10.sp,
                        fontFamily = FontFamily.Monospace
                    )
                }
                val metaLine = buildString {
                    if (log.plateOcr != null && !log.vehicleMmc.isNullOrBlank()) {
                        append(log.vehicleMmc).append("  •  ")
                    }
                    if (!log.notes.isNullOrBlank() && log.plateOcr != null) {
                        append(log.notes).append("  •  ")
                    }
                    append(formattedDate)
                }
                Text(
                    text       = metaLine,
                    color      = Color.Gray,
                    fontSize   = 9.sp,
                    fontFamily = FontFamily.Monospace
                )
            }
            val thumb = cropBitmap
            if (thumb != null) {
                androidx.compose.foundation.Image(
                    bitmap             = thumb.asImageBitmap(),
                    contentDescription = "Plate crop",
                    modifier           = Modifier
                        .size(width = 72.dp, height = 36.dp)
                        .background(Color.Black, RoundedCornerShape(4.dp))
                )
            } else {
                Text(text = formattedDate, color = Color.Gray, fontSize = 10.sp, fontFamily = FontFamily.Monospace)
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Plate bounding box overlay
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun BoundingBoxOverlay(
    boxes: List<PlateBox>,
    frameAspectRatio: Float = 0.75f,
    modifier: Modifier = Modifier
) {
    val green = JoltColors.bboxGreen
    Canvas(modifier = modifier) {
        if (boxes.isEmpty() || size.width <= 0f || size.height <= 0f) return@Canvas
        val canvasAspect = size.width / size.height
        val safeFrameAspect = if (frameAspectRatio > 0f) frameAspectRatio else canvasAspect
        val renderW: Float
        val renderH: Float
        if (safeFrameAspect > canvasAspect) {
            renderH = size.height
            renderW = size.height * safeFrameAspect
        } else {
            renderW = size.width
            renderH = size.width / safeFrameAspect
        }
        val offsetX = (size.width - renderW) / 2f
        val offsetY = (size.height - renderH) / 2f

        boxes.forEach { box ->
            val l = offsetX + box.left   * renderW
            val t = offsetY + box.top    * renderH
            val r = offsetX + box.right  * renderW
            val b = offsetY + box.bottom * renderH
            drawRect(
                color    = green,
                topLeft  = Offset(l, t),
                size     = GeoSize(r - l, b - t),
                style    = Stroke(width = 3.dp.toPx())
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CameraX preview composable (R12 preserved)
// ─────────────────────────────────────────────────────────────────────────────

@OptIn(ExperimentalCamera2Interop::class)
@Composable
fun CameraPreviewWidget(
    lifecycleOwner: androidx.lifecycle.LifecycleOwner,
    cameraProviderFuture: ListenableFuture<ProcessCameraProvider>,
    analyzer: TelephotoAnalyzer?
) {
    val context = LocalContext.current

    val imageAnalysis = remember {
        ImageAnalysis.Builder()
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
    }

    DisposableEffect(imageAnalysis, analyzer) {
        if (analyzer != null) {
            imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(context), analyzer)
            Log.i("CameraPreviewWidget", "ImageAnalysis analyzer attached.")
        }
        onDispose { imageAnalysis.clearAnalyzer() }
    }

    AndroidView(
        factory = { ctx ->
            val previewView = PreviewView(ctx)
            val executor    = ContextCompat.getMainExecutor(ctx)

            cameraProviderFuture.addListener({
                try {
                    val cameraProvider = cameraProviderFuture.get()

                    val preview = Preview.Builder().build().apply {
                        setSurfaceProvider(previewView.surfaceProvider)
                    }

                    preview.targetRotation = android.view.Surface.ROTATION_0
                    imageAnalysis.targetRotation = android.view.Surface.ROTATION_0

                    val telephotoSelector = findTelephotoCameraSelector(cameraProvider)
                    try {
                        cameraProvider.unbindAll()
                        val camera = cameraProvider.bindToLifecycle(
                            lifecycleOwner, telephotoSelector, preview, imageAnalysis
                        )
                        camera.cameraControl.setZoomRatio(5.0f)
                        Log.i("CameraPreviewWidget", "Bound to telephoto lens at 5× zoom.")
                    } catch (e: Exception) {
                        Log.w("CameraPreviewWidget", "Telephoto binding failed — falling back to default rear camera.")
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageAnalysis
                        )
                    }
                } catch (e: Exception) {
                    Log.e("CameraPreviewWidget", "CameraProvider setup failed: ${e.localizedMessage}", e)
                }
            }, executor)

            previewView
        },
        modifier = Modifier.fillMaxSize()
    )
}

@OptIn(ExperimentalCamera2Interop::class)
private fun findTelephotoCameraSelector(cameraProvider: ProcessCameraProvider): CameraSelector {
    for (cameraInfo in cameraProvider.availableCameraInfos) {
        val cam2Info    = Camera2CameraInfo.from(cameraInfo)
        val lensFacing  = cam2Info.getCameraCharacteristic(CameraCharacteristics.LENS_FACING)
        if (lensFacing == CameraCharacteristics.LENS_FACING_BACK) {
            val focalLengths = cam2Info.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
            if (focalLengths != null && focalLengths.any { it >= 7.0f }) {
                return cameraInfo.cameraSelector
            }
        }
    }
    return CameraSelector.DEFAULT_BACK_CAMERA
}
