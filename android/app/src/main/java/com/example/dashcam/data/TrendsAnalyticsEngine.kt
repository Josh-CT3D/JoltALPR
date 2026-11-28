package com.ct3d.jolt.data

import kotlin.math.max

/**
 * Phase 3 Private Trend Analytics Engine (`Jolt_Customer_Requirements.txt` §11).
 *
 * Computes:
 *  - Raw incident counts AND normalized incident rates (`flagged / observed encounters`) by Make,
 *    Body Style, and Color.
 *  - Jack's Central CT Make Watchlist (`Honda`, `Acura`, `Infiniti`) vs all makes.
 *  - Jack's `"White SUV"` hypothesis spotlight (`White SUV` incident rate vs overall rate).
 *  - Behavior trigger frequency breakdown & Repeat-Offender vehicle leaderboard.
 */
object TrendsAnalyticsEngine {

    data class CategoryRateRow(
        val label: String,
        val flaggedCount: Int,
        val observedEncounters: Int,
        val ratePct: Float,
        val isWatchlist: Boolean = false
    )

    data class WhiteSuvSpotlight(
        val flaggedWhiteSuvs: Int,
        val observedWhiteSuvs: Int,
        val whiteSuvRatePct: Float,
        val allOtherFlagged: Int,
        val allOtherObserved: Int,
        val allOtherRatePct: Float,
        val relativeMultiplier: Float
    )

    data class TriggerStatRow(
        val tag: BehaviorTag,
        val count: Int,
        val sharePct: Float
    )

    data class TrendsReport(
        val totalBadIncidents: Int,
        val totalAngels: Int,
        val totalSightings: Int,
        val uniqueFlaggedPlates: Int,
        val peerSharedIncidents: Int,
        val whiteSuvSpotlight: WhiteSuvSpotlight,
        val makeLeaderboard: List<CategoryRateRow>,
        val watchlistRows: List<CategoryRateRow>,
        val bodyLeaderboard: List<CategoryRateRow>,
        val colorLeaderboard: List<CategoryRateRow>,
        val triggerBreakdown: List<TriggerStatRow>,
        val repeatOffenders: List<VehicleReputation>
    )

    fun buildReport(
        allLogs: List<DriverLog>,
        encounters: List<AppearanceEncounter>
    ): TrendsReport {
        val badLogs = allLogs.filter { it.rating == "BAD" }
        val angelLogs = allLogs.filter { it.rating == "ANGEL" }
        val sightingLogs = allLogs.filter { it.rating == "SIGHTING" }

        val encMap = encounters.associate { (it.categoryType to it.categoryKey) to it.encounterCount }

        val makeFlags = HashMap<String, Int>()
        val bodyFlags = HashMap<String, Int>()
        val colorFlags = HashMap<String, Int>()
        var whiteSuvFlags = 0
        val tagCounts = HashMap<BehaviorTag, Int>()

        for (row in badLogs) {
            val app = PartialMatchEngine.resolveStoredAppearance(listOf(row))
            app.make?.let { mk -> makeFlags[mk] = (makeFlags[mk] ?: 0) + 1 }
            app.bodyStyle?.let { bd -> bodyFlags[bd] = (bodyFlags[bd] ?: 0) + 1 }
            app.color?.let { cl -> colorFlags[cl] = (colorFlags[cl] ?: 0) + 1 }
            if (app.color.equals("White", ignoreCase = true) && app.bodyStyle.equals("SUV", ignoreCase = true)) {
                whiteSuvFlags++
            }
            for (tag in BehaviorTag.parseList(row.behaviorTags)) {
                tagCounts[tag] = (tagCounts[tag] ?: 0) + 1
            }
        }

        // Build Make Leaderboard (combining flagged makes + encountered makes + watchlist makes)
        val allMakes = (
            makeFlags.keys +
                encounters.filter { it.categoryType == "MAKE" }.map { it.categoryKey } +
                VehicleAppearance.WATCHLIST_MAKES
            ).distinct()

        val makeRows = allMakes.map { mk ->
            val flg = makeFlags[mk] ?: 0
            val obs = max(flg, encMap["MAKE" to mk] ?: 0)
            val rate = if (obs > 0) (flg.toFloat() / obs.toFloat()) * 100f else 0f
            CategoryRateRow(
                label = mk,
                flaggedCount = flg,
                observedEncounters = obs,
                ratePct = rate,
                isWatchlist = mk in VehicleAppearance.WATCHLIST_MAKES
            )
        }.filter { it.flaggedCount > 0 || it.observedEncounters > 0 || it.isWatchlist }
            .sortedWith(compareByDescending<CategoryRateRow> { it.flaggedCount }.thenByDescending { it.ratePct })

        val watchlistRows = VehicleAppearance.WATCHLIST_MAKES.map { mk ->
            makeRows.firstOrNull { it.label.equals(mk, ignoreCase = true) }
                ?: CategoryRateRow(mk, 0, 0, 0f, isWatchlist = true)
        }

        // Body Style Leaderboard
        val allBodies = (bodyFlags.keys + encounters.filter { it.categoryType == "BODY" }.map { it.categoryKey }).distinct()
        val bodyRows = allBodies.map { bd ->
            val flg = bodyFlags[bd] ?: 0
            val obs = max(flg, encMap["BODY" to bd] ?: 0)
            val rate = if (obs > 0) (flg.toFloat() / obs.toFloat()) * 100f else 0f
            CategoryRateRow(bd, flg, obs, rate)
        }.sortedWith(compareByDescending<CategoryRateRow> { it.flaggedCount }.thenByDescending { it.ratePct })

        // Color Leaderboard
        val allColors = (colorFlags.keys + encounters.filter { it.categoryType == "COLOR" }.map { it.categoryKey }).distinct()
        val colorRows = allColors.map { cl ->
            val flg = colorFlags[cl] ?: 0
            val obs = max(flg, encMap["COLOR" to cl] ?: 0)
            val rate = if (obs > 0) (flg.toFloat() / obs.toFloat()) * 100f else 0f
            CategoryRateRow(cl, flg, obs, rate)
        }.sortedWith(compareByDescending<CategoryRateRow> { it.flaggedCount }.thenByDescending { it.ratePct })

        // White SUV Hypothesis Spotlight
        val obsWhiteSuvs = max(whiteSuvFlags, encMap["COLOR_BODY" to "White SUV"] ?: 0)
        val whiteSuvRate = if (obsWhiteSuvs > 0) (whiteSuvFlags.toFloat() / obsWhiteSuvs.toFloat()) * 100f else 0f
        val totalBodyObserved = bodyRows.sumOf { it.observedEncounters }
        val otherFlags = (badLogs.size - whiteSuvFlags).coerceAtLeast(0)
        val otherObs = max(otherFlags, (totalBodyObserved - obsWhiteSuvs).coerceAtLeast(0))
        val otherRate = if (otherObs > 0) (otherFlags.toFloat() / otherObs.toFloat()) * 100f else 0f
        val mult = if (otherRate > 0.01f && whiteSuvRate > 0f) whiteSuvRate / otherRate else 0f

        val spotlight = WhiteSuvSpotlight(
            flaggedWhiteSuvs = whiteSuvFlags,
            observedWhiteSuvs = obsWhiteSuvs,
            whiteSuvRatePct = whiteSuvRate,
            allOtherFlagged = otherFlags,
            allOtherObserved = otherObs,
            allOtherRatePct = otherRate,
            relativeMultiplier = mult
        )

        // Trigger Breakdown
        val totalTagMentions = tagCounts.values.sum().coerceAtLeast(1)
        val triggerRows = tagCounts.entries
            .sortedByDescending { it.value }
            .map { (tag, cnt) ->
                TriggerStatRow(tag, cnt, (cnt.toFloat() / totalTagMentions.toFloat()) * 100f)
            }

        // Repeat Offenders (plates with >=2 BAD incidents or >=1 BAD + >=1 SIGHTING)
        val byPlate = allLogs.filter { !it.plateNormalized.isNullOrBlank() }
            .groupBy { it.plateNormalized!! }
        val repeatOffenders = byPlate.values
            .mapNotNull { rows -> computeVehicleReputation(rows, matchConfidence = 1.0f) }
            .filter { it.badIncidentCount >= 2 || (it.badIncidentCount >= 1 && it.sightingCount >= 1) }
            .sortedWith(
                compareByDescending<VehicleReputation> { it.effectiveSeverity.rank }
                    .thenByDescending { it.badIncidentCount }
                    .thenByDescending { it.sightingCount }
            )
            .take(15)

        val uniquePlates = badLogs.mapNotNull { it.plateNormalized?.takeIf { p -> p.isNotBlank() } }.toSet().size
        val peerShared = allLogs.count { it.reporterId != "LOCAL" && it.rating in setOf("BAD", "ANGEL") }

        return TrendsReport(
            totalBadIncidents = badLogs.size,
            totalAngels = angelLogs.size,
            totalSightings = sightingLogs.size,
            uniqueFlaggedPlates = uniquePlates,
            peerSharedIncidents = peerShared,
            whiteSuvSpotlight = spotlight,
            makeLeaderboard = makeRows,
            watchlistRows = watchlistRows,
            bodyLeaderboard = bodyRows,
            colorLeaderboard = colorRows,
            triggerBreakdown = triggerRows,
            repeatOffenders = repeatOffenders
        )
    }
}
