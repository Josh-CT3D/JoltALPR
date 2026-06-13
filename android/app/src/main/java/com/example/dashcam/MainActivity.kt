package com.ct3d.jolt

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.camera2.CameraCharacteristics
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
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
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size as GeoSize
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.lifecycleScope
import android.graphics.BitmapFactory
import android.graphics.RectF
import com.ct3d.jolt.camera.TelephotoAnalyzer
import com.ct3d.jolt.data.DriverLog
import com.ct3d.jolt.ui.DashcamViewModel
import com.google.common.util.concurrent.ListenableFuture
import kotlinx.coroutines.launch
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
 * Main Activity — Jolt ALPR Dashcam
 *
 * UI: landscape split-screen (camera left | OSMDroid map right) +
 * large red FLAG BAD DRIVER button + bottom navigation tabs.
 *
 * Voice commands removed — physical button only.
 */
class MainActivity : ComponentActivity() {

    private val viewModel: DashcamViewModel by viewModels()
    private lateinit var cameraExecutor: ExecutorService
    private var telephotoAnalyzer: TelephotoAnalyzer? = null

    private val requiredPermissions = arrayOf(
        Manifest.permission.CAMERA,
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.ACCESS_COARSE_LOCATION
    )

    private val permissionsLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val cameraOk   = permissions[Manifest.permission.CAMERA] ?: false
        val locationOk = permissions[Manifest.permission.ACCESS_FINE_LOCATION] ?: false
        if (cameraOk && locationOk) {
            setupServices()
        } else {
            Toast.makeText(this, "Camera and GPS permissions are required.", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Configure OSMDroid tile cache directory
        Configuration.getInstance().load(
            applicationContext,
            applicationContext.getSharedPreferences("osmdroid", Context.MODE_PRIVATE)
        )

        if (allPermissionsGranted()) setupServices()
        else permissionsLauncher.launch(requiredPermissions)

        setContent {
            JoltTheme {
                JoltApp(
                    viewModel = viewModel,
                    getAnalyzer = { telephotoAnalyzer }
                )
            }
        }
    }

    private fun allPermissionsGranted() = requiredPermissions.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private fun setupServices() {
        Log.i(TAG, "Initializing ALPR pipeline analyzer...")
        try {
            telephotoAnalyzer = TelephotoAnalyzer(applicationContext, lifecycleScope).also { analyzer ->
                lifecycleScope.launch {
                    analyzer.pipelineState.collect { state ->
                        viewModel.updateActiveDetection(
                            state.activePlateocr,
                            state.activeVehicleMmc,
                            state.plateBoxes,
                            state.lastPlateCrop
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
// Theme
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun JoltTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            primary    = Color(0xFFFF1744), // Red — FLAG button accent
            secondary  = Color(0xFF00E676), // Green — active detections
            tertiary   = Color(0xFF00B0FF), // Blue — info / MMC
            background = Color(0xFF0A0A0A),
            surface    = Color(0xFF1A1A1A),
            onSurface  = Color.White
        ),
        content = content
    )
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
    val logList            by viewModel.logsList.collectAsState()
    val gpsLocation        by viewModel.currentLocation.collectAsState()
    val bannerNotification by viewModel.notificationMessage.collectAsState()
    val badDriverAlert     by viewModel.knownBadDriverAlert.collectAsState()

    // Wrap everything in a Box so the alert border can float above the Scaffold
    Box(modifier = Modifier.fillMaxSize()) {
    Scaffold(
        bottomBar = {
            NavigationBar(
                containerColor = Color(0xFF111111),
                tonalElevation = 0.dp,
                modifier = Modifier.height(80.dp)
            ) {
                JoltNavItem(0, selectedTab, { selectedTab = 0 }, Icons.Default.Videocam,      "Main")
                JoltNavItem(1, selectedTab, { selectedTab = 1 }, Icons.Default.Map,           "Map")
                JoltNavItem(2, selectedTab, { selectedTab = 2 }, Icons.Default.List,          "History")
                JoltNavItem(3, selectedTab, { selectedTab = 3 }, Icons.Default.FileDownload,  "Export")
                JoltNavItem(4, selectedTab, { selectedTab = 4 }, Icons.Default.ManageSearch,  "Manage")
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
                    activeOcr          = activeOcr,
                    activeMmc          = activeMmc,
                    plateBoxes         = plateBoxes,
                    bannerNotification = bannerNotification,
                    onDismissNotification = viewModel::dismissNotification,
                    onFlagBadDriver    = viewModel::flagBadDriver,
                    getAnalyzer        = getAnalyzer
                )
                1 -> JoltMapScreen(
                    logs            = logList,
                    currentLocation = gpsLocation
                )
                2 -> HistoryScreen(logList = logList)
                3 -> ExportScreen(logList = logList)
                4 -> ManageScreen(
                    logList     = logList,
                    onDeleteLog = viewModel::deleteLogItem,
                    onClearAll  = viewModel::clearLogs
                )
            }
        }
    } // end Scaffold

    // Known Bad Driver Alert — yellow pulsing border overlaid above everything
    if (badDriverAlert != null) {
        PulsingAlertBorder(
            plate     = badDriverAlert!!.plateOcr ?: "Unknown",
            onDismiss = viewModel::dismissAlert
        )
    }

    } // end outer Box
}

// ─────────────────────────────────────────────────────────────────────────────
// Known Bad Driver Alert — yellow pulsing screen border
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun PulsingAlertBorder(plate: String, onDismiss: () -> Unit) {
    // Auto-dismiss after 15 seconds
    LaunchedEffect(plate) {
        kotlinx.coroutines.delay(15_000L)
        onDismiss()
    }

    val infiniteTransition = rememberInfiniteTransition(label = "alert_pulse")
    val borderAlpha by infiniteTransition.animateFloat(
        initialValue = 0.35f,
        targetValue  = 1.0f,
        animationSpec = infiniteRepeatable(
            animation  = tween(450, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "border_alpha"
    )

    Box(modifier = Modifier.fillMaxSize()) {
        // Full-perimeter yellow pulsing border
        Box(
            modifier = Modifier
                .fillMaxSize()
                .border(10.dp, Color(0xFFFFD700).copy(alpha = borderAlpha))
        )

        // Warning card at top-center
        Card(
            shape    = RoundedCornerShape(8.dp),
            colors   = CardDefaults.cardColors(containerColor = Color(0xFFFFD700)),
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = 6.dp)
        ) {
            Row(
                modifier          = Modifier.padding(horizontal = 16.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector        = Icons.Default.Warning,
                    contentDescription = null,
                    tint               = Color.Black,
                    modifier           = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(10.dp))
                Text(
                    text       = "⚠  CAUTION — KNOWN BAD DRIVER  |  PLATE: $plate",
                    color      = Color.Black,
                    fontWeight = FontWeight.ExtraBold,
                    fontSize   = 14.sp,
                    fontFamily = FontFamily.Monospace
                )
                Spacer(modifier = Modifier.width(10.dp))
                IconButton(onClick = onDismiss, modifier = Modifier.size(28.dp)) {
                    Icon(Icons.Default.Close, contentDescription = "Dismiss", tint = Color.Black)
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
            selectedIconColor   = Color(0xFFFF1744),
            selectedTextColor   = Color(0xFFFF1744),
            unselectedIconColor = Color.Gray,
            unselectedTextColor = Color.Gray,
            indicatorColor      = Color(0xFF2A0A0F)
        )
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Screen: camera (left) | map (right) | FLAG button
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun MainScreen(
    activeOcr: String?,
    activeMmc: String?,
    plateBoxes: List<RectF>,
    bannerNotification: String?,
    onDismissNotification: () -> Unit,
    onFlagBadDriver: () -> Unit,
    getAnalyzer: () -> TelephotoAnalyzer?
) {
    val context        = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    Column(modifier = Modifier.fillMaxSize()) {

        // Notification banner (slides in at top)
        AnimatedVisibility(visible = bannerNotification != null) {
            Surface(
                color = Color(0xFF00B0FF),
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

        // ── Full-width camera preview — fills ~78% of remaining height ──
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

            // Green bounding boxes drawn over detected plates
            BoundingBoxOverlay(
                boxes    = plateBoxes,
                modifier = Modifier.fillMaxSize()
            )

            // Lens indicator chip (top-left)
            Box(
                modifier = Modifier
                    .align(Alignment.TopStart)
                    .padding(8.dp)
                    .background(Color.Black.copy(alpha = 0.65f), RoundedCornerShape(4.dp))
                    .padding(horizontal = 8.dp, vertical = 4.dp)
            ) {
                Text(
                    text       = "5× TELEPHOTO",
                    color      = Color(0xFF00E676),
                    fontSize   = 10.sp,
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.Bold
                )
            }

            // OCR / MMC result chip (bottom of camera view)
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
                                color      = Color(0xFF00E676),
                                fontSize   = 9.sp,
                                fontFamily = FontFamily.Monospace,
                                textAlign  = TextAlign.Center
                            )
                        } else {
                            Text(
                                text       = activeMmc ?: "",
                                color      = Color(0xFF00B0FF),
                                fontSize   = 16.sp,
                                fontWeight = FontWeight.Bold,
                                textAlign  = TextAlign.Center
                            )
                            Text(
                                text       = "MMC FALLBACK",
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

        // ── FLAG BAD DRIVER button — fills ~22% of remaining height ──
        Button(
            onClick  = onFlagBadDriver,
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
                .padding(horizontal = 16.dp, vertical = 10.dp),
            colors    = ButtonDefaults.buttonColors(containerColor = Color(0xFFFF1744)),
            shape     = RoundedCornerShape(14.dp),
            elevation = ButtonDefaults.buttonElevation(defaultElevation = 8.dp)
        ) {
            Icon(
                imageVector        = Icons.Default.Flag,
                contentDescription = null,
                tint               = Color.White,
                modifier           = Modifier.size(28.dp)
            )
            Spacer(modifier = Modifier.width(12.dp))
            Text(
                text          = "FLAG BAD DRIVER",
                color         = Color.White,
                fontSize      = 20.sp,
                fontWeight    = FontWeight.ExtraBold,
                letterSpacing = 2.sp
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Map tab — full-screen OSMDroid map with flagged pins
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun JoltMapScreen(
    logs: List<DriverLog>,
    currentLocation: android.location.Location?
) {
    JoltMapView(
        modifier        = Modifier.fillMaxSize(),
        logs            = logs,
        currentLocation = currentLocation
    )
}

@Composable
fun JoltMapView(
    modifier: Modifier,
    logs: List<DriverLog>,
    currentLocation: android.location.Location?
) {
    val lifecycleOwner = LocalLifecycleOwner.current
    var mapViewRef by remember { mutableStateOf<MapView?>(null) }
    // Once the user pans or zooms the map manually, stop snapping to GPS so the
    // map doesn't fight them. Re-centers only until the first touch event.
    var userHasPanned by remember { mutableStateOf(false) }

    // Honour MapView pause/resume with Activity lifecycle
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            when (event) {
                Lifecycle.Event.ON_RESUME -> mapViewRef?.onResume()
                Lifecycle.Event.ON_PAUSE  -> mapViewRef?.onPause()
                else -> {}
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    AndroidView(
        factory = { ctx ->
            MapView(ctx).also { mapView ->
                mapViewRef = mapView
                mapView.setTileSource(TileSourceFactory.MAPNIK)
                mapView.setMultiTouchControls(true)
                mapView.controller.setZoom(15.0)
                // Default center — will animate to real GPS when available
                mapView.controller.setCenter(GeoPoint(37.7749, -122.4194))
                mapView.setOnTouchListener { v, _ ->
                    userHasPanned = true
                    v.performClick()
                    false // let OSMDroid handle the event normally
                }
            }
        },
        update = { mapView ->
            mapView.overlays.clear()
            val fmt = SimpleDateFormat("MM/dd HH:mm", Locale.getDefault())

            // Red pins for every flagged capture
            logs.forEach { log ->
                val marker = Marker(mapView).apply {
                    position = GeoPoint(log.latitude, log.longitude)
                    title    = log.plateOcr ?: log.vehicleMmc ?: "Flagged Vehicle"
                    snippet  = fmt.format(Date(log.timestamp))
                }
                mapView.overlays.add(marker)
            }

            // Follow GPS position until the user manually moves the map
            if (!userHasPanned) {
                currentLocation?.let { loc ->
                    mapView.controller.animateTo(GeoPoint(loc.latitude, loc.longitude))
                }
            }
            mapView.invalidate()
        },
        modifier = modifier
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// History tab
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun HistoryScreen(logList: List<DriverLog>) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text(
            text       = "CAPTURE HISTORY",
            fontWeight = FontWeight.Bold,
            fontSize   = 13.sp,
            color      = Color.Gray,
            fontFamily = FontFamily.Monospace,
            modifier   = Modifier.padding(bottom = 12.dp)
        )
        if (logList.isEmpty()) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text(
                    text      = "No captures yet.\nFlag a bad driver on the main screen to see records here.",
                    color     = Color.Gray,
                    textAlign = TextAlign.Center,
                    fontSize  = 13.sp
                )
            }
        } else {
            LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(logList) { log -> DriverLogItemWidget(log = log) }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Export tab
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun ExportScreen(logList: List<DriverLog>) {
    val context = LocalContext.current

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text       = "EXPORT DATA",
            fontWeight = FontWeight.Bold,
            fontSize   = 16.sp,
            color      = Color.Gray,
            fontFamily = FontFamily.Monospace,
            modifier   = Modifier.padding(bottom = 24.dp)
        )

        Button(
            onClick  = { exportToCsv(context, logList) },
            modifier = Modifier.fillMaxWidth().height(54.dp),
            colors   = ButtonDefaults.buttonColors(containerColor = Color(0xFF1565C0)),
            shape    = RoundedCornerShape(10.dp)
        ) {
            Icon(Icons.Default.FileDownload, contentDescription = null)
            Spacer(modifier = Modifier.width(10.dp))
            Text("Export Records (CSV)", fontWeight = FontWeight.Bold, fontSize = 14.sp)
        }

        Spacer(modifier = Modifier.height(10.dp))
        Text(
            text     = "${logList.size} flagged capture${if (logList.size != 1) "s" else ""} in database",
            color    = Color.Gray,
            fontSize = 12.sp
        )

        Spacer(modifier = Modifier.height(24.dp))

        // Training data export card (Phase 6 — shown as informational for now)
        Card(
            colors   = CardDefaults.cardColors(containerColor = Color(0xFF1A1A1A)),
            shape    = RoundedCornerShape(10.dp),
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    text       = "Training Data Export",
                    fontWeight = FontWeight.SemiBold,
                    color      = Color.White,
                    fontSize   = 14.sp
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text     = "Enable Training Mode (Phase 6) to collect annotated plate images.\n" +
                               "Exports a Roboflow-compatible ZIP (images/ + labels/) for model fine-tuning.",
                    color    = Color.Gray,
                    fontSize = 11.sp
                )
            }
        }
    }
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
                stream.write("ID,Timestamp,Latitude,Longitude,PlateOCR,VehicleMMC,Rating,Battery\n".toByteArray())
                logs.forEach { log ->
                    val line = "${log.id},${log.timestamp},${log.latitude},${log.longitude}," +
                               "${log.plateOcr ?: ""},${log.vehicleMmc ?: ""},${log.rating},${log.batteryLevel}\n"
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
            text             = { Text("This will permanently delete all ${logList.size} flagged captures.") },
            confirmButton    = {
                TextButton(onClick = { onClearAll(); showClearDialog = false }) {
                    Text("Delete All", color = Color(0xFFFF1744))
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
                    colors  = ButtonDefaults.buttonColors(containerColor = Color(0xFFFF1744)),
                    contentPadding = PaddingValues(horizontal = 14.dp, vertical = 6.dp)
                ) {
                    Text("Clear All", color = Color.White, fontSize = 12.sp)
                }
            }
        }

        if (logList.isEmpty()) {
            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text("No records to manage.", color = Color.Gray, fontSize = 13.sp)
            }
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
        colors   = CardDefaults.cardColors(containerColor = Color(0xFF1E1E1E)),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier          = Modifier.padding(12.dp).fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text       = log.plateOcr ?: log.vehicleMmc ?: "Unidentified",
                    color      = Color.White,
                    fontWeight = FontWeight.Bold,
                    fontSize   = 13.sp,
                    fontFamily = if (log.plateOcr != null) FontFamily.Monospace else FontFamily.Default
                )
                Text(
                    text     = "GPS: [${String.format("%.4f", log.latitude)}, ${String.format("%.4f", log.longitude)}]",
                    color    = Color.Gray,
                    fontSize = 9.sp,
                    fontFamily = FontFamily.Monospace
                )
                Text(text = fmt, color = Color.Gray, fontSize = 9.sp)
            }
            IconButton(onClick = onDelete) {
                Icon(Icons.Default.Delete, contentDescription = "Delete", tint = Color(0xFFFF1744))
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared UI: history log item card
// ─────────────────────────────────────────────────────────────────────────────

@Composable
fun DriverLogItemWidget(log: DriverLog) {
    val ratingColor = if (log.rating == "BAD") Color(0xFFFF1744) else Color(0xFF00E676)
    val formattedDate = remember(log.timestamp) {
        SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date(log.timestamp))
    }
    val cropBitmap = remember(log.plateCropPath) {
        log.plateCropPath?.let { path ->
            try { BitmapFactory.decodeFile(path) } catch (e: Exception) { null }
        }
    }
    Card(
        shape    = RoundedCornerShape(8.dp),
        colors   = CardDefaults.cardColors(containerColor = Color(0xFF1E1E1E)),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier          = Modifier.padding(12.dp).fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text       = log.rating,
                color      = ratingColor,
                fontWeight = FontWeight.Bold,
                fontSize   = 11.sp,
                fontFamily = FontFamily.Monospace,
                modifier   = Modifier.width(36.dp)
            )
            Column(modifier = Modifier.weight(1f).padding(horizontal = 12.dp)) {
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
                        text       = log.vehicleMmc ?: "Unidentified",
                        color      = Color(0xFF00B0FF),
                        fontWeight = FontWeight.SemiBold,
                        fontSize   = 12.sp
                    )
                }
                Text(
                    text     = "GPS: [${String.format("%.4f", log.latitude)}, ${String.format("%.4f", log.longitude)}]",
                    color    = Color.Gray,
                    fontSize = 9.sp,
                    fontFamily = FontFamily.Monospace
                )
            }
            if (cropBitmap != null) {
                androidx.compose.foundation.Image(
                    bitmap             = cropBitmap.asImageBitmap(),
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
fun BoundingBoxOverlay(boxes: List<RectF>, modifier: Modifier = Modifier) {
    val green = Color(0xFF00E676)
    Canvas(modifier = modifier) {
        boxes.forEach { box ->
            val l = box.left   * size.width
            val t = box.top    * size.height
            val r = box.right  * size.width
            val b = box.bottom * size.height
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
// CameraX preview composable (unchanged logic, package updated)
// ─────────────────────────────────────────────────────────────────────────────

@OptIn(ExperimentalCamera2Interop::class)
@Composable
fun CameraPreviewWidget(
    lifecycleOwner: androidx.lifecycle.LifecycleOwner,
    cameraProviderFuture: ListenableFuture<ProcessCameraProvider>,
    analyzer: TelephotoAnalyzer?
) {
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

                    val imageAnalysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()

                    if (analyzer != null) {
                        imageAnalysis.setAnalyzer(executor, analyzer)
                    }

                    // Auto-rotate CameraX with device orientation
                    val orientationEventListener = object : android.view.OrientationEventListener(ctx) {
                        override fun onOrientationChanged(orientation: Int) {
                            if (orientation == ORIENTATION_UNKNOWN) return
                            val rotation = when (orientation) {
                                in 45 until 135  -> android.view.Surface.ROTATION_270
                                in 135 until 225 -> android.view.Surface.ROTATION_180
                                in 225 until 315 -> android.view.Surface.ROTATION_90
                                else             -> android.view.Surface.ROTATION_0
                            }
                            try {
                                preview.targetRotation       = rotation
                                imageAnalysis.targetRotation = rotation
                            } catch (e: Exception) {
                                Log.w("CameraPreviewWidget", "Rotation update failed: ${e.localizedMessage}")
                            }
                        }
                    }

                    previewView.addOnAttachStateChangeListener(object : android.view.View.OnAttachStateChangeListener {
                        override fun onViewAttachedToWindow(v: android.view.View)  { orientationEventListener.enable() }
                        override fun onViewDetachedFromWindow(v: android.view.View) { orientationEventListener.disable() }
                    })

                    val telephotoSelector = findTelephotoCameraSelector(cameraProvider)
                    try {
                        cameraProvider.unbindAll()
                        val camera = cameraProvider.bindToLifecycle(
                            lifecycleOwner, telephotoSelector, preview, imageAnalysis
                        )
                        // 5× zoom targets the Pixel 10 telephoto lens
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
            // Pixel 10 telephoto focal length > 7.0mm; wide lens ~6.5mm
            if (focalLengths != null && focalLengths.any { it >= 7.0f }) {
                return cameraInfo.cameraSelector
            }
        }
    }
    return CameraSelector.DEFAULT_BACK_CAMERA
}
