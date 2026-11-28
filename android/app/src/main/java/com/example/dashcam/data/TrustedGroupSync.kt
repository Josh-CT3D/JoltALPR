package com.ct3d.jolt.data

import android.content.ContentValues
import android.content.Context
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

/**
 * Phase 3 Closed Trusted-Group Sync (`Jolt_Customer_Requirements.txt` §12).
 *
 * Allows 3–4 trusted household / commuter phones to export and import vehicle reports as a
 * self-contained RFC-8259 `.json` bundle (via Quick Share, Drive, or SMS/email) WITHOUT any server:
 *  - Privacy-first (§12): GPS coordinates (`latitude`, `longitude`) are stripped (`0.0, 0.0`) on
 *    export so sharing vehicle warnings never exposes a group member's live commute locations.
 *  - Provenance & Advisory Weight (§12): each unit has a persistent short unit ID; imported peer
 *    records retain `reporterId = "Peer:<unitId>"` and advisory confidence (`0.90f`).
 *  - Uses a self-contained pure-Kotlin RFC-8259 JSON encoder/decoder so unit tests run directly
 *    on the JVM without Android `org.json` stub issues.
 */
class TrustedGroupSync(private val context: Context) {

    private val appContext = context.applicationContext
    private val prefs = appContext.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun getOrCreateUnitId(): String {
        val existing = prefs.getString(KEY_UNIT_ID, null)
        if (!existing.isNullOrBlank()) return existing
        val generated = "Unit-" + UUID.randomUUID().toString().uppercase(Locale.US).take(4)
        prefs.edit().putString(KEY_UNIT_ID, generated).apply()
        return generated
    }

    data class ImportResult(
        val importedCount: Int,
        val skippedDuplicates: Int,
        val sourceUnit: String
    )

    fun exportGroupBundleToDownloads(allLogs: List<DriverLog>): String? {
        val localShareable = allLogs.filter {
            it.reporterId == "LOCAL" && (it.rating == "BAD" || it.rating == "ANGEL") &&
                (!it.plateNormalized.isNullOrBlank() || !it.vehicleMmc.isNullOrBlank())
        }
        if (localShareable.isEmpty()) return null

        val unitId = getOrCreateUnitId()
        val payload = serializeBundleJson(unitId, localShareable)
        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val fileName = "jolt_group_${unitId}_$ts.json"

        val values = ContentValues().apply {
            put(MediaStore.Downloads.DISPLAY_NAME, fileName)
            put(MediaStore.Downloads.MIME_TYPE, "application/json")
        }
        val uri = appContext.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
            ?: return null

        return try {
            val stream = appContext.contentResolver.openOutputStream(uri)
                ?: throw IOException("Cannot open output stream for $uri")
            stream.bufferedWriter(Charsets.UTF_8).use { it.write(payload) }
            Log.i(TAG, "Exported ${localShareable.size} records into trusted-group bundle $fileName")
            fileName
        } catch (e: Exception) {
            Log.e(TAG, "Trusted-group export failed: ${e.localizedMessage}", e)
            try { appContext.contentResolver.delete(uri, null, null) } catch (_: Exception) {}
            null
        }
    }

    suspend fun importGroupBundleFromUri(uri: Uri, dao: DriverLogDao): ImportResult {
        val rawJson = appContext.contentResolver.openInputStream(uri)?.bufferedReader(Charsets.UTF_8)?.use {
            it.readText()
        } ?: throw IOException("Could not read selected bundle file.")

        val myUnitId = getOrCreateUnitId()
        val parsed = parseBundleJson(rawJson, myUnitId)
        var imported = 0
        var skipped = 0

        for (row in parsed.rows) {
            val dupCount = dao.countMatchingImportedLog(
                reporterId = row.reporterId,
                timestamp = row.timestamp,
                plateNormOrEmpty = row.plateNormalized ?: ""
            )
            if (dupCount > 0) {
                skipped++
            } else {
                dao.safeInsertLog(row)
                imported++
            }
        }
        return ImportResult(imported, skipped, parsed.sourceUnit)
    }

    companion object {
        private const val TAG = "TrustedGroupSync"
        private const val PREFS_NAME = "jolt_trusted_group_prefs"
        private const val KEY_UNIT_ID = "unit_id"
        private const val SCHEMA_VERSION = 1

        data class ParsedBundle(val sourceUnit: String, val rows: List<DriverLog>)

        internal fun serializeBundleJson(unitId: String, logs: List<DriverLog>): String {
            val sb = StringBuilder(512 + logs.size * 256)
            sb.append("{\n")
            sb.append("  \"schemaVersion\": ").append(SCHEMA_VERSION).append(",\n")
            sb.append("  \"unitId\": ").append(jsonString(unitId)).append(",\n")
            sb.append("  \"exportedAtMs\": ").append(System.currentTimeMillis()).append(",\n")
            sb.append("  \"records\": [\n")
            logs.forEachIndexed { idx, log ->
                sb.append("    {")
                sb.append("\"rating\": ").append(jsonString(log.rating)).append(", ")
                sb.append("\"plateOcr\": ").append(jsonString(log.plateOcr)).append(", ")
                sb.append("\"plateNormalized\": ").append(jsonString(log.plateNormalized)).append(", ")
                sb.append("\"vehicleMmc\": ").append(jsonString(log.vehicleMmc)).append(", ")
                sb.append("\"vehicleColor\": ").append(jsonString(log.vehicleColor)).append(", ")
                sb.append("\"vehicleMake\": ").append(jsonString(log.vehicleMake)).append(", ")
                sb.append("\"vehicleBodyStyle\": ").append(jsonString(log.vehicleBodyStyle)).append(", ")
                sb.append("\"behaviorTags\": ").append(jsonString(log.behaviorTags)).append(", ")
                sb.append("\"severity\": ").append(jsonString(log.severity)).append(", ")
                sb.append("\"notes\": ").append(jsonString(log.notes)).append(", ")
                sb.append("\"timestamp\": ").append(log.timestamp)
                // Deliberately omit latitude/longitude & local file paths for group privacy (§12)
                sb.append("}")
                if (idx < logs.lastIndex) sb.append(",")
                sb.append("\n")
            }
            sb.append("  ]\n}")
            return sb.toString()
        }

        internal fun parseBundleJson(rawJson: String, localUnitId: String): ParsedBundle {
            val root = SimpleJsonReader(rawJson).readObject()
            val sourceUnit = (root["unitId"] as? String)?.ifBlank { "Peer" } ?: "Peer"
            val reporterTag = if (sourceUnit == localUnitId) "LOCAL" else "Peer:$sourceUnit"
            val records = (root["records"] as? List<*>) ?: emptyList<Any?>()
            val out = ArrayList<DriverLog>(records.size)

            for (entry in records) {
                @Suppress("UNCHECKED_CAST")
                val obj = entry as? Map<String, Any?> ?: continue
                val rating = (obj["rating"] as? String) ?: "BAD"
                if (rating != "BAD" && rating != "ANGEL") continue

                val plateOcr = (obj["plateOcr"] as? String)?.takeIf { it.isNotBlank() }
                val plateNorm = (obj["plateNormalized"] as? String)?.takeIf { it.isNotBlank() }
                    ?: plateOcr?.let { normalizePlate(it) }
                val ts = (obj["timestamp"] as? Number)?.toLong() ?: System.currentTimeMillis()

                out.add(
                    DriverLog(
                        id = 0,
                        rating = rating,
                        plateOcr = plateOcr,
                        plateNormalized = plateNorm,
                        vehicleMmc = (obj["vehicleMmc"] as? String)?.takeIf { it.isNotBlank() },
                        timestamp = ts,
                        latitude = 0.0,
                        longitude = 0.0,
                        batteryLevel = 100,
                        plateCropPath = null,
                        behaviorTags = (obj["behaviorTags"] as? String)?.takeIf { it.isNotBlank() },
                        severity = (obj["severity"] as? String)?.takeIf { it.isNotBlank() },
                        matchConfidence = if (reporterTag == "LOCAL") 1.0f else 0.90f,
                        notes = (obj["notes"] as? String)?.takeIf { it.isNotBlank() },
                        vehicleColor = (obj["vehicleColor"] as? String)?.takeIf { it.isNotBlank() },
                        vehicleMake = (obj["vehicleMake"] as? String)?.takeIf { it.isNotBlank() },
                        vehicleBodyStyle = (obj["vehicleBodyStyle"] as? String)?.takeIf { it.isNotBlank() },
                        reporterId = reporterTag
                    )
                )
            }
            return ParsedBundle(sourceUnit, out)
        }

        private fun jsonString(value: String?): String {
            if (value == null) return "null"
            val sb = StringBuilder(value.length + 4)
            sb.append('"')
            for (ch in value) {
                when (ch) {
                    '"'  -> sb.append("\\\"")
                    '\\' -> sb.append("\\\\")
                    '\n' -> sb.append("\\n")
                    '\r' -> sb.append("\\r")
                    '\t' -> sb.append("\\t")
                    else -> if (ch.code < 0x20) {
                        sb.append("\\u%04x".format(Locale.US, ch.code))
                    } else {
                        sb.append(ch)
                    }
                }
            }
            sb.append('"')
            return sb.toString()
        }

        /** Minimal RFC-8259 recursive-descent JSON reader (Objects, Arrays, Strings, Numbers, Booleans, null). */
        private class SimpleJsonReader(private val s: String) {
            private var pos = 0

            private fun skipWs() {
                while (pos < s.length && s[pos].isWhitespace()) pos++
            }

            fun readValue(): Any? {
                skipWs()
                if (pos >= s.length) throw IllegalArgumentException("Unexpected end of JSON")
                return when (val c = s[pos]) {
                    '{' -> readObject()
                    '[' -> readArray()
                    '"' -> readString()
                    't' -> { expect("true"); true }
                    'f' -> { expect("false"); false }
                    'n' -> { expect("null"); null }
                    '-', in '0'..'9' -> readNumber()
                    else -> throw IllegalArgumentException("Unexpected char '$c' at $pos")
                }
            }

            fun readObject(): Map<String, Any?> {
                skipWs()
                expect("{")
                val map = LinkedHashMap<String, Any?>()
                skipWs()
                if (pos < s.length && s[pos] == '}') {
                    pos++
                    return map
                }
                while (true) {
                    skipWs()
                    val key = readString()
                    skipWs()
                    expect(":")
                    val v = readValue()
                    map[key] = v
                    skipWs()
                    if (pos < s.length && s[pos] == ',') {
                        pos++
                    } else {
                        expect("}")
                        break
                    }
                }
                return map
            }

            private fun readArray(): List<Any?> {
                skipWs()
                expect("[")
                val list = ArrayList<Any?>()
                skipWs()
                if (pos < s.length && s[pos] == ']') {
                    pos++
                    return list
                }
                while (true) {
                    list.add(readValue())
                    skipWs()
                    if (pos < s.length && s[pos] == ',') {
                        pos++
                    } else {
                        expect("]")
                        break
                    }
                }
                return list
            }

            private fun readString(): String {
                expect("\"")
                val sb = StringBuilder()
                while (pos < s.length) {
                    val c = s[pos++]
                    if (c == '"') return sb.toString()
                    if (c == '\\') {
                        if (pos >= s.length) throw IllegalArgumentException("Unterminated escape")
                        when (val esc = s[pos++]) {
                            '"', '\\', '/' -> sb.append(esc)
                            'b' -> sb.append('\b')
                            'f' -> sb.append('\u000C')
                            'n' -> sb.append('\n')
                            'r' -> sb.append('\r')
                            't' -> sb.append('\t')
                            'u' -> {
                                val hex = s.substring(pos, (pos + 4).coerceAtMost(s.length))
                                pos += 4
                                sb.append(hex.toInt(16).toChar())
                            }
                            else -> sb.append(esc)
                        }
                    } else {
                        sb.append(c)
                    }
                }
                throw IllegalArgumentException("Unterminated JSON string")
            }

            private fun readNumber(): Number {
                val start = pos
                while (pos < s.length && (s[pos] in '0'..'9' || s[pos] in "-+.eE")) pos++
                val token = s.substring(start, pos)
                return if ('.' in token || 'e' in token || 'E' in token) token.toDouble() else token.toLong()
            }

            private fun expect(lit: String) {
                if (!s.regionMatches(pos, lit, 0, lit.length)) {
                    throw IllegalArgumentException("Expected '$lit' at pos $pos")
                }
                pos += lit.length
            }
        }
    }
}
