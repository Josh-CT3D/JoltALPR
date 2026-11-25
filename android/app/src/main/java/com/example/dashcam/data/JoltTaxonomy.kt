package com.ct3d.jolt.data

/**
 * Phase 2 Jolt Behavior Vocabulary, Severity Scale, Angel Status & Vehicle Reputation Calculator.
 *
 * Implements the 8-category KISS trigger list, 4-tier hazard color spectrum + parallel Angel mark,
 * and permanent multi-incident reputation rules from Jack Kwon's Jolt requirements.
 */
enum class JoltSeverity(
    val rank: Int,
    val displayLabel: String,
    val spokenLabel: String,
    val defensiveAction: String
) {
    ANGEL(0, "ANGEL", "Angel", "Known courteous vehicle"),
    YELLOW(1, "YELLOW", "Yellow", "Observe"),
    ORANGE(2, "ORANGE", "Orange", "Create space"),
    RED(3, "RED", "Red", "Keep clear — do not stay beside"),
    MAROON(4, "MAROON", "Maroon", "Disengage and maximize distance");

    companion object {
        fun fromCode(code: String?): JoltSeverity? =
            entries.firstOrNull { it.name.equals(code?.trim(), ignoreCase = true) }
    }
}

enum class BehaviorTag(
    val displayLabel: String,
    val shortHudCode: String,
    val spokenPhrase: String,
    val defaultSeverity: JoltSeverity,
    val voicePhrases: List<String>
) {
    WEAVING(
        displayLabel = "Weaving",
        shortHudCode = "WEAVER",
        spokenPhrase = "weaver",
        defaultSeverity = JoltSeverity.ORANGE,
        voicePhrases = listOf("weaving", "weaver", "drifting")
    ),
    THROTTLE(
        displayLabel = "Unsteady Throttle",
        shortHudCode = "THROTTLE",
        spokenPhrase = "unsteady throttle",
        defaultSeverity = JoltSeverity.YELLOW,
        voicePhrases = listOf("braking", "throttle", "unsteady throttle", "pedal illiterate", "brake checking")
    ),
    CUTTING(
        displayLabel = "Cutting",
        shortHudCode = "LINE CUTTER",
        spokenPhrase = "line cutter",
        defaultSeverity = JoltSeverity.RED,
        voicePhrases = listOf("cutting", "cutter", "line cutter", "cut me off", "made me brake")
    ),
    TAILGATING(
        displayLabel = "Tailgating",
        shortHudCode = "TAILGATER",
        spokenPhrase = "tailgater",
        defaultSeverity = JoltSeverity.RED,
        voicePhrases = listOf("tailgating", "tailgater", "riding")
    ),
    SPEED(
        displayLabel = "Speed",
        shortHudCode = "SPEED",
        spokenPhrase = "unstable speed",
        defaultSeverity = JoltSeverity.YELLOW,
        voicePhrases = listOf("speed", "speeding", "too fast", "too slow")
    ),
    NO_LANE(
        displayLabel = "No Lane / Signal",
        shortHudCode = "NO LANE",
        spokenPhrase = "no lane control",
        defaultSeverity = JoltSeverity.ORANGE,
        voicePhrases = listOf("no lane", "cant hold lane", "no blinker", "rubbernecking", "lane blocker")
    ),
    NO_PLATE(
        displayLabel = "No / Obscured Plate",
        shortHudCode = "NO PLATE",
        spokenPhrase = "obscured plate",
        defaultSeverity = JoltSeverity.ORANGE,
        voicePhrases = listOf("no plate", "obscured plate", "dark plate", "tinted")
    ),
    ASSHOLE(
        displayLabel = "Hostile / Asshole",
        shortHudCode = "HOSTILE",
        spokenPhrase = "known asshole",
        defaultSeverity = JoltSeverity.RED,
        voicePhrases = listOf("asshole", "hostile", "road rage")
    );

    companion object {
        const val MAX_TAGS_PER_INCIDENT = 2

        fun serialize(tags: List<BehaviorTag>): String? =
            tags.distinct().take(MAX_TAGS_PER_INCIDENT).takeIf { it.isNotEmpty() }
                ?.joinToString(",") { it.name }

        fun parseList(serialized: String?): List<BehaviorTag> {
            if (serialized.isNullOrBlank()) return emptyList()
            return serialized.split(',')
                .mapNotNull { token -> entries.firstOrNull { it.name.equals(token.trim(), ignoreCase = true) } }
                .distinct()
        }

        fun defaultSeverityFor(tags: List<BehaviorTag>, explicitOverride: JoltSeverity? = null): JoltSeverity {
            if (explicitOverride != null && explicitOverride != JoltSeverity.ANGEL) return explicitOverride
            if (tags.isEmpty()) return JoltSeverity.ORANGE
            // Brake-checking or 2+ red tags still stays RED on a single incident; Maroon requires
            // explicit override or repeated RED incidents across encounters (Requirements §6).
            return tags.maxBy { it.defaultSeverity.rank }.defaultSeverity
        }
    }
}

/**
 * Calculated cumulative history for a recognized vehicle.
 * Keeps severity, incident count (pips), Angel goodwill, and match confidence strictly separate
 * (Requirements §7: "Severity, frequency, recency, and identification confidence must never be
 * collapsed into one unexplained score").
 */
data class VehicleReputation(
    val plateDisplay: String,
    val plateNormalized: String,
    val vehicleMmc: String?,
    val effectiveSeverity: JoltSeverity,
    val badIncidentCount: Int,
    val angelCount: Int,
    val sightingCount: Int,
    val topTags: List<BehaviorTag>,
    val matchConfidence: Float,
    val latestIncidentTimestamp: Long
) {
    val hasAngelHalo: Boolean get() = angelCount > 0
    val isPureAngel: Boolean get() = badIncidentCount == 0 && angelCount > 0

    val primaryHudBehavior: String
        get() = when {
            isPureAngel -> "COURTEOUS DRIVER"
            topTags.isNotEmpty() -> topTags.take(2).joinToString(" + ") { it.shortHudCode }
            else -> "FLAGGED VEHICLE"
        }

    /** Short spoken callout while driving (Requirements §9 & Repeat Encounter: "Red. Weaver."). */
    fun buildSpokenAlert(): String {
        if (isPureAngel) return "Angel vehicle ahead."
        val sev = effectiveSeverity.spokenLabel
        val behavior = topTags.firstOrNull()?.spokenPhrase ?: "flagged driver"
        val repeatPrefix = if (badIncidentCount >= 2) "Repeat " else ""
        val countSuffix = if (badIncidentCount >= 2) ". $badIncidentCount prior flags" else ""
        val angelSuffix = if (hasAngelHalo) ". Also marked angel" else ""
        val suspectPrefix = if (matchConfidence < 0.95f) "Suspected " else ""
        return "$suspectPrefix$sev. $repeatPrefix$behavior$countSuffix$angelSuffix."
    }
}

/**
 * Pure Reputation Summary builder over all Room rows matching a normalized plate.
 * Records never expire or decay (Requirements: "Records remain permanently unless manually deleted").
 */
fun computeVehicleReputation(logs: List<DriverLog>, matchConfidence: Float = 1.0f): VehicleReputation? {
    if (logs.isEmpty()) return null
    val badLogs = logs.filter { it.rating == "BAD" }
    val angelLogs = logs.filter { it.rating == "ANGEL" }
    val sightingLogs = logs.filter { it.rating == "SIGHTING" }

    if (badLogs.isEmpty() && angelLogs.isEmpty()) return null

    val newest = (badLogs + angelLogs).maxBy { it.timestamp }
    val plateDisplay = newest.plateOcr ?: logs.firstNotNullOfOrNull { it.plateOcr } ?: "UNKNOWN"
    val plateNorm = newest.plateNormalized ?: normalizePlate(plateDisplay)
    val mmc = logs.firstNotNullOfOrNull { it.vehicleMmc }

    // Rank behavior tags by frequency across BAD incidents, breaking ties by severity rank.
    val tagCounts = mutableMapOf<BehaviorTag, Int>()
    for (row in badLogs) {
        for (tag in BehaviorTag.parseList(row.behaviorTags)) {
            tagCounts[tag] = (tagCounts[tag] ?: 0) + 1
        }
    }
    val rankedTags = tagCounts.entries
        .sortedWith(compareByDescending<Map.Entry<BehaviorTag, Int>> { it.value }.thenByDescending { it.key.defaultSeverity.rank })
        .map { it.key }

    val effectiveSeverity = if (badLogs.isEmpty()) {
        JoltSeverity.ANGEL
    } else {
        val highestRecorded = badLogs.mapNotNull { row ->
            JoltSeverity.fromCode(row.severity)
                ?: BehaviorTag.parseList(row.behaviorTags).maxByOrNull { it.defaultSeverity.rank }?.defaultSeverity
        }.maxByOrNull { it.rank } ?: JoltSeverity.ORANGE

        when {
            highestRecorded == JoltSeverity.MAROON -> JoltSeverity.MAROON
            // Multiple RED incidents across encounters escalate to MAROON (Requirements §6)
            badLogs.size >= 2 && highestRecorded == JoltSeverity.RED -> JoltSeverity.MAROON
            // 3+ ORANGE incidents escalate to RED
            badLogs.size >= 3 && highestRecorded == JoltSeverity.ORANGE -> JoltSeverity.RED
            // Multiple YELLOW incidents escalate to ORANGE (Requirements §6)
            badLogs.size >= 2 && highestRecorded == JoltSeverity.YELLOW -> JoltSeverity.ORANGE
            else -> highestRecorded
        }
    }

    return VehicleReputation(
        plateDisplay            = plateDisplay,
        plateNormalized         = plateNorm,
        vehicleMmc              = mmc,
        effectiveSeverity       = effectiveSeverity,
        badIncidentCount        = badLogs.size,
        angelCount              = angelLogs.size,
        sightingCount           = sightingLogs.size,
        topTags                 = rankedTags,
        matchConfidence         = matchConfidence,
        latestIncidentTimestamp = newest.timestamp
    )
}
