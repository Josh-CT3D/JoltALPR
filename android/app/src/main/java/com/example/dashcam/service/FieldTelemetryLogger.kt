package com.ct3d.jolt.service

import android.app.ActivityManager
import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.provider.MediaStore
import android.util.Log
import com.ct3d.jolt.camera.TelephotoAnalyzer
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * On-device Field Telemetry & Soak Logger for drive testing without a tethered PC.
 *
 * Appends structured tab-separated records to `filesDir/telemetry/telemetry_events.log` on a
 * dedicated background single-thread executor (so logging never blocks CameraX, `jolt-pipeline`,
 * or the UI thread, and survives app restarts or crashes).
 *
 * When exported back at the shop via [exportReportToDownloads], writes a self-contained `.txt`
 * report into `Downloads/jolt_telemetry_<timestamp>.txt` with an automatically-computed
 * Executive Summary Header (latency p50/p95/max, busy-drops, O4 3fps readiness check, thermal
 * status/headroom & battery temp °C, Java/native heap soak delta, OCR hit rate, and O9 MMC
 * confidence distribution) followed by the raw chronological event trace.
 */
class FieldTelemetryLogger(
    private val context: Context,
    private val batteryMonitor: BatteryMonitor = BatteryMonitor(context)
) {
    private val appContext = context.applicationContext
    private val prefs = appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    private val telemetryDir = File(appContext.filesDir, "telemetry").also { it.mkdirs() }
    private val logFile = File(telemetryDir, LOG_FILE_NAME)

    private val writerExecutor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "jolt-telemetry-writer").apply { isDaemon = true }
    }

    // PowerManager.getThermalHeadroom() is rate-limited by Android to >=10s between calls.
    private val lastHeadroomQueryMs = AtomicLong(0L)
    @Volatile
    private var cachedThermalHeadroom: Float = Float.NaN

    private val eventCount = AtomicInteger(countExistingLines())

    fun isPersistedEnabled(): Boolean = prefs.getBoolean(KEY_ENABLED, false)

    fun setPersistedEnabled(enabled: Boolean) {
        prefs.edit().putBoolean(KEY_ENABLED, enabled).apply()
    }

    fun currentEventCount(): Int = eventCount.get()

    fun recordSessionMarker(marker: String, analysisIntervalMs: Long, gpsSpeedMph: Float?) {
        val ts = formatIso(System.currentTimeMillis())
        val device = "${Build.MANUFACTURER} ${Build.MODEL} (SDK ${Build.VERSION.SDK_INT})"
        val line = listOf(
            "SESSION",
            ts,
            "event=$marker",
            "device=${sanitize(device)}",
            "intervalMs=$analysisIntervalMs",
            "speedMph=${formatFloat(gpsSpeedMph, 1)}"
        ).joinToString("\t")
        enqueueLine(line)
        // Always capture an immediate system health snapshot alongside session start/stop.
        recordHealthSnapshot(gpsSpeedMph)
    }

    /**
     * Captures thermal throttling state, thermal headroom, battery % & °C, Java heap, native
     * Bitmap/TFLite heap, and system available RAM. Call periodically (e.g. every 15s).
     */
    fun recordHealthSnapshot(gpsSpeedMph: Float?) {
        val now = System.currentTimeMillis()
        val ts = formatIso(now)

        val powerManager = appContext.getSystemService(Context.POWER_SERVICE) as? PowerManager
        val thermalCode = powerManager?.currentThermalStatus ?: -1
        val thermalName = thermalStatusName(thermalCode)

        val prevHeadroomAt = lastHeadroomQueryMs.get()
        if (powerManager != null && now - prevHeadroomAt >= 11_000L &&
            lastHeadroomQueryMs.compareAndSet(prevHeadroomAt, now)
        ) {
            cachedThermalHeadroom = try {
                powerManager.getThermalHeadroom(10)
            } catch (_: Exception) {
                Float.NaN
            }
        }

        val battery = batteryMonitor.getSnapshot()
        val rt = Runtime.getRuntime()
        val javaUsedMb = (rt.totalMemory() - rt.freeMemory()) / BYTES_PER_MB
        val javaMaxMb = rt.maxMemory() / BYTES_PER_MB
        val nativeAllocMb = Debug.getNativeHeapAllocatedSize() / BYTES_PER_MB

        val memInfo = ActivityManager.MemoryInfo()
        val am = appContext.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        am?.getMemoryInfo(memInfo)
        val availRamMb = memInfo.availMem / BYTES_PER_MB

        val line = listOf(
            "HEALTH",
            ts,
            "thermal=$thermalName($thermalCode)",
            "headroom=${formatFloat(cachedThermalHeadroom.takeIf { !it.isNaN() }, 3)}",
            "batPct=${battery.levelPct}",
            "batTempC=${formatFloat(battery.temperatureC, 1)}",
            "charging=${battery.isCharging}",
            "javaMb=${"%.1f".format(Locale.US, javaUsedMb)}/${"%.0f".format(Locale.US, javaMaxMb)}",
            "nativeMb=${"%.1f".format(Locale.US, nativeAllocMb)}",
            "availRamMb=${"%.0f".format(Locale.US, availRamMb)}",
            "lowMem=${memInfo.lowMemory}",
            "speedMph=${formatFloat(gpsSpeedMph, 1)}"
        ).joinToString("\t")
        enqueueLine(line)
    }

    fun recordFrame(frame: TelephotoAnalyzer.FrameTelemetry, gpsSpeedMph: Float?) {
        val ts = formatIso(frame.timestampMs)
        val tokensStr = if (frame.ocrTokens.isEmpty()) "-" else frame.ocrTokens.joinToString(",")
        val line = listOf(
            "FRAME",
            ts,
            "dtMs=${frame.frameDeltaMs}",
            "drops=${frame.busyDrops}",
            "totalMs=${frame.totalMs}",
            "yoloMs=${frame.yoloMs}",
            "ocrMs=${frame.ocrMs}",
            "mmcMs=${frame.mmcMs}",
            "gpu=${if (frame.yoloGpu) 1 else 0}/${if (frame.mmcGpu) 1 else 0}",
            "boxes=${frame.plateBoxCount}",
            "boxConf=${"%.3f".format(Locale.US, frame.topPlateConf)}",
            "ocrTokens=${sanitize(tokensStr)}",
            "consensus=${sanitize(frame.consensusPlate ?: "-")}",
            "mmcTrig=${frame.mmcTrigger}",
            "mmcPub=${sanitize(frame.mmcPublished ?: "-")}",
            "mmcConf=${"%.3f".format(Locale.US, frame.mmcTop1Conf)}",
            "mmcTop3=${sanitize(frame.mmcTop3 ?: "-")}",
            "speedMph=${formatFloat(gpsSpeedMph, 1)}"
        ).joinToString("\t")
        enqueueLine(line)
    }

    fun recordAction(action: String, detail: String, gpsSpeedMph: Float? = null) {
        val ts = formatIso(System.currentTimeMillis())
        val line = listOf(
            "ACTION",
            ts,
            "action=${sanitize(action)}",
            "detail=${sanitize(detail)}",
            "speedMph=${formatFloat(gpsSpeedMph, 1)}"
        ).joinToString("\t")
        enqueueLine(line)
    }

    /**
     * Flushes pending writes and exports `Downloads/jolt_telemetry_<yyyyMMdd_HHmmss>.txt`
     * containing an Executive Summary Header + all raw event lines. Call off the main thread.
     */
    fun exportReportToDownloads(): String? {
        flushWriter()
        if (!logFile.exists() || logFile.length() == 0L) {
            Log.w(TAG, "No telemetry events to export.")
            return null
        }

        val lines = try {
            logFile.useLines { it.toList() }
        } catch (e: Exception) {
            Log.e(TAG, "Failed reading telemetry log: ${e.localizedMessage}", e)
            return null
        }
        if (lines.isEmpty()) return null

        val header = buildSummaryHeader(lines)
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val fileName = "jolt_telemetry_$ts.txt"
        val values = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, fileName)
            put(MediaStore.Downloads.MIME_TYPE, "text/plain")
        }

        val uri = appContext.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            ?: return null

        return try {
            val stream = appContext.contentResolver.openOutputStream(uri)
                ?: throw IOException("Could not open MediaStore stream for $uri")
            stream.bufferedWriter(Charsets.UTF_8).use { writer ->
                writer.write(header)
                writer.write("\n")
                for (l in lines) {
                    writer.write(l)
                    writer.write("\n")
                }
            }
            Log.i(TAG, "Exported field telemetry report (${lines.size} lines) to Downloads/$fileName")
            fileName
        } catch (e: Exception) {
            Log.e(TAG, "Telemetry export failed: ${e.localizedMessage}", e)
            try { appContext.contentResolver.delete(uri, null, null) } catch (_: Exception) {}
            null
        }
    }

    fun clearAll(): Int {
        flushWriter()
        if (logFile.exists()) logFile.delete()
        eventCount.set(0)
        return 0
    }

    fun release() {
        try {
            writerExecutor.shutdown()
            writerExecutor.awaitTermination(500, TimeUnit.MILLISECONDS)
        } catch (_: Exception) {}
    }

    private fun enqueueLine(line: String) {
        eventCount.incrementAndGet()
        try {
            writerExecutor.execute {
                try {
                    telemetryDir.mkdirs()
                    FileOutputStream(logFile, true).bufferedWriter(Charsets.UTF_8).use { out ->
                        out.write(line)
                        out.write("\n")
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Telemetry write failed: ${e.localizedMessage}")
                }
            }
        } catch (_: Exception) {}
    }

    private fun flushWriter() {
        try {
            writerExecutor.submit {}.get(2, TimeUnit.SECONDS)
        } catch (_: Exception) {}
    }

    private fun countExistingLines(): Int {
        if (!logFile.exists()) return 0
        return try {
            logFile.useLines { seq -> seq.count() }
        } catch (_: Exception) {
            0
        }
    }

    companion object {
        private const val TAG = "FieldTelemetryLogger"
        private const val PREFS_NAME = "jolt_telemetry_prefs"
        private const val KEY_ENABLED = "telemetry_enabled"
        private const val LOG_FILE_NAME = "telemetry_events.log"
        private const val BYTES_PER_MB = 1048576.0

        private fun formatIso(epochMs: Long): String =
            SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ", Locale.US).format(Date(epochMs))

        private fun formatFloat(v: Float?, decimals: Int): String =
            if (v == null || v.isNaN()) "N/A" else "%.${decimals}f".format(Locale.US, v)

        private fun sanitize(raw: String): String =
            raw.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').trim()

        internal fun thermalStatusName(status: Int): String = when (status) {
            PowerManager.THERMAL_STATUS_NONE      -> "NONE"
            PowerManager.THERMAL_STATUS_LIGHT     -> "LIGHT"
            PowerManager.THERMAL_STATUS_MODERATE  -> "MODERATE"
            PowerManager.THERMAL_STATUS_SEVERE    -> "SEVERE"
            PowerManager.THERMAL_STATUS_CRITICAL  -> "CRITICAL"
            PowerManager.THERMAL_STATUS_EMERGENCY -> "EMERGENCY"
            PowerManager.THERMAL_STATUS_SHUTDOWN  -> "SHUTDOWN"
            else                                  -> "UNKNOWN"
        }

        /**
         * Pure summary generator over raw TSV telemetry lines — produces an instant diagnostic
         * digest for Phase 5 (O9 MMC), Phase 6, and Phase 7 (J4/O4 thermal & 3fps decision).
         */
        internal fun buildSummaryHeader(lines: List<String>): String {
            var frameCount = 0
            var totalBusyDrops = 0
            val totalLatencies = mutableListOf<Long>()
            val yoloLatencies = mutableListOf<Long>()
            val ocrLatencies = mutableListOf<Long>()
            val mmcLatencies = mutableListOf<Long>()

            var boxesFoundFrames = 0
            var ocrTokenFrames = 0
            var consensusFrames = 0
            var mmcCenterCalls = 0
            var mmcAnchoredCalls = 0
            var mmcPublishedCount = 0
            val mmcAnchoredConfs = mutableListOf<Float>()
            val mmcCenterConfs = mutableListOf<Float>()

            var worstThermalCode = -1
            var worstThermalLabel = "N/A"
            var maxHeadroom = Float.NaN
            var minBatTemp = Float.POSITIVE_INFINITY
            var maxBatTemp = Float.NEGATIVE_INFINITY
            var firstBatPct: Int? = null
            var lastBatPct: Int? = null

            var firstJavaMb: Double? = null
            var lastJavaMb: Double? = null
            var maxJavaMb = 0.0
            var firstNativeMb: Double? = null
            var lastNativeMb: Double? = null
            var maxNativeMb = 0.0
            var maxSpeedMph = 0f

            var firstTimestamp: String? = null
            var lastTimestamp: String? = null

            for (line in lines) {
                val cols = line.split('\t')
                if (cols.size < 3) continue
                val type = cols[0]
                val ts = cols[1]
                if (firstTimestamp == null) firstTimestamp = ts
                lastTimestamp = ts

                val kv = HashMap<String, String>(cols.size)
                for (i in 2 until cols.size) {
                    val eq = cols[i].indexOf('=')
                    if (eq > 0) {
                        kv[cols[i].substring(0, eq)] = cols[i].substring(eq + 1)
                    }
                }

                kv["speedMph"]?.toFloatOrNull()?.let { if (it > maxSpeedMph) maxSpeedMph = it }

                when (type) {
                    "FRAME" -> {
                        frameCount++
                        totalBusyDrops += kv["drops"]?.toIntOrNull() ?: 0
                        kv["totalMs"]?.toLongOrNull()?.let { totalLatencies.add(it) }
                        kv["yoloMs"]?.toLongOrNull()?.let { yoloLatencies.add(it) }
                        kv["ocrMs"]?.toLongOrNull()?.let { if (it > 0) ocrLatencies.add(it) }
                        kv["mmcMs"]?.toLongOrNull()?.let { if (it > 0) mmcLatencies.add(it) }

                        val boxes = kv["boxes"]?.toIntOrNull() ?: 0
                        if (boxes > 0) boxesFoundFrames++
                        if ((kv["ocrTokens"] ?: "-") != "-") ocrTokenFrames++
                        if ((kv["consensus"] ?: "-") != "-") consensusFrames++

                        val trig = kv["mmcTrig"] ?: "NONE"
                        val mmcConf = kv["mmcConf"]?.toFloatOrNull() ?: 0f
                        if (trig == "NO_PLATE_CENTER") {
                            mmcCenterCalls++
                            if (mmcConf > 0f) mmcCenterConfs.add(mmcConf)
                        } else if (trig == "OCR_FAIL_ANCHORED") {
                            mmcAnchoredCalls++
                            if (mmcConf > 0f) mmcAnchoredConfs.add(mmcConf)
                        }
                        if ((kv["mmcPub"] ?: "-") != "-") mmcPublishedCount++
                    }
                    "HEALTH" -> {
                        val thermalRaw = kv["thermal"] ?: ""
                        val code = thermalRaw.substringAfter('(', "").substringBefore(')', "").toIntOrNull() ?: -1
                        if (code > worstThermalCode) {
                            worstThermalCode = code
                            worstThermalLabel = thermalRaw
                        }
                        kv["headroom"]?.toFloatOrNull()?.let {
                            if (maxHeadroom.isNaN() || it > maxHeadroom) maxHeadroom = it
                        }
                        kv["batTempC"]?.toFloatOrNull()?.let {
                            if (it < minBatTemp) minBatTemp = it
                            if (it > maxBatTemp) maxBatTemp = it
                        }
                        kv["batPct"]?.toIntOrNull()?.let {
                            if (it >= 0) {
                                if (firstBatPct == null) firstBatPct = it
                                lastBatPct = it
                            }
                        }
                        kv["javaMb"]?.substringBefore('/')?.toDoubleOrNull()?.let {
                            if (firstJavaMb == null) firstJavaMb = it
                            lastJavaMb = it
                            if (it > maxJavaMb) maxJavaMb = it
                        }
                        kv["nativeMb"]?.toDoubleOrNull()?.let {
                            if (firstNativeMb == null) firstNativeMb = it
                            lastNativeMb = it
                            if (it > maxNativeMb) maxNativeMb = it
                        }
                    }
                }
            }

            totalLatencies.sort()
            val avgTotal = if (totalLatencies.isNotEmpty()) totalLatencies.average() else 0.0
            val p50Total = percentile(totalLatencies, 0.50)
            val p95Total = percentile(totalLatencies, 0.95)
            val maxTotal = totalLatencies.lastOrNull() ?: 0L
            val avgYolo = if (yoloLatencies.isNotEmpty()) yoloLatencies.average() else 0.0
            val avgOcr = if (ocrLatencies.isNotEmpty()) ocrLatencies.average() else 0.0
            val avgMmc = if (mmcLatencies.isNotEmpty()) mmcLatencies.average() else 0.0

            val nativeDelta = if (firstNativeMb != null && lastNativeMb != null) lastNativeMb!! - firstNativeMb!! else 0.0
            val readyFor3Fps = frameCount >= 20 &&
                p95Total in 1..290 &&
                worstThermalCode <= PowerManager.THERMAL_STATUS_LIGHT &&
                nativeDelta < 35.0

            return buildString {
                appendLine("========================================================================")
                appendLine("JOLT ALPR — FIELD TELEMETRY & SOAK SUMMARY")
                appendLine("========================================================================")
                appendLine("Span:            ${firstTimestamp ?: "N/A"}  ->  ${lastTimestamp ?: "N/A"}")
                appendLine("Total events:    ${lines.size}  |  Analyzed frames: $frameCount  |  Busy drops: $totalBusyDrops")
                appendLine("Max GPS speed:   ${"%.1f".format(Locale.US, maxSpeedMph)} mph")
                appendLine()
                appendLine("--- PIPELINE LATENCY ---")
                appendLine("Total frame ms:  avg=${"%.1f".format(Locale.US, avgTotal)}ms  p50=${p50Total}ms  p95=${p95Total}ms  max=${maxTotal}ms")
                appendLine("Stage avg ms:    YOLO=${"%.1f".format(Locale.US, avgYolo)}ms  OCR(active)=${"%.1f".format(Locale.US, avgOcr)}ms  MMC(active)=${"%.1f".format(Locale.US, avgMmc)}ms")
                appendLine()
                appendLine("--- THERMAL & BATTERY SOAK (J4) ---")
                appendLine("Peak thermal:    $worstThermalLabel  |  Peak headroom(10s): ${formatFloat(maxHeadroom.takeIf { !it.isNaN() }, 3)}")
                val tempRange = if (minBatTemp.isFinite()) "${"%.1f".format(Locale.US, minBatTemp)}°C .. ${"%.1f".format(Locale.US, maxBatTemp)}°C" else "N/A"
                appendLine("Battery temp:    $tempRange  |  Battery level: ${firstBatPct ?: "N/A"}% -> ${lastBatPct ?: "N/A"}%")
                appendLine()
                appendLine("--- MEMORY SOAK (S1 #8) ---")
                appendLine("Java heap MB:    start=${fmtD(firstJavaMb)}  end=${fmtD(lastJavaMb)}  peak=${fmtD(maxJavaMb)}")
                appendLine("Native heap MB:  start=${fmtD(firstNativeMb)}  end=${fmtD(lastNativeMb)}  peak=${fmtD(maxNativeMb)}  (delta=${"%+.1f".format(Locale.US, nativeDelta)} MB)")
                appendLine()
                appendLine("--- DETECTION, OCR & MMC ACCURACY (O9) ---")
                appendLine("Plate detected:  $boxesFoundFrames / $frameCount frames (${pct(boxesFoundFrames, frameCount)})")
                appendLine("OCR token read:  $ocrTokenFrames frames  |  Consensus published: $consensusFrames frames")
                val totalMmc = mmcCenterCalls + mmcAnchoredCalls
                appendLine("MMC triggered:   $totalMmc frames (anchored=$mmcAnchoredCalls, center=$mmcCenterCalls)  |  Published (>=0.25): $mmcPublishedCount")
                if (mmcAnchoredConfs.isNotEmpty()) {
                    appendLine("MMC anchored:    avgConf=${"%.3f".format(Locale.US, mmcAnchoredConfs.average())}  maxConf=${"%.3f".format(Locale.US, mmcAnchoredConfs.max())}")
                }
                if (mmcCenterConfs.isNotEmpty()) {
                    appendLine("MMC center:      avgConf=${"%.3f".format(Locale.US, mmcCenterConfs.average())}  maxConf=${"%.3f".format(Locale.US, mmcCenterConfs.max())}")
                }
                appendLine()
                appendLine("--- O4 3-FPS BUMP RECOMMENDATION ---")
                appendLine(
                    if (frameCount < 20) "INSUFFICIENT DATA (<20 frames) — run a 30-min drive soak first."
                    else if (readyFor3Fps) "PASS — p95=${p95Total}ms (<290ms), thermal<=LIGHT, native heap stable. Safe to test 3 FPS (333ms)."
                    else "HOLD AT 2 FPS — check p95 latency (${p95Total}ms), peak thermal ($worstThermalLabel), or native delta (${"%+.1f".format(Locale.US, nativeDelta)} MB)."
                )
                appendLine("========================================================================")
            }
        }

        private fun percentile(sorted: List<Long>, p: Double): Long {
            if (sorted.isEmpty()) return 0L
            val idx = ((sorted.size - 1) * p).toInt().coerceIn(0, sorted.lastIndex)
            return sorted[idx]
        }

        private fun fmtD(v: Double?): String =
            if (v == null) "N/A" else "%.1f".format(Locale.US, v)

        private fun pct(part: Int, total: Int): String =
            if (total <= 0) "0.0%" else "%.1f%%".format(Locale.US, part * 100.0 / total)
    }
}
