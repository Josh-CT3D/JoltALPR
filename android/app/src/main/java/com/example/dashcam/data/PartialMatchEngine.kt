package com.ct3d.jolt.data

import kotlin.math.abs

/**
 * Phase 3 Tier 2 (Partial / Fuzzy Plate + Appearance) & Tier 3 (Obscured-Plate Appearance)
 * Vehicle Recognition Engine (`Jolt_Customer_Requirements.txt` § Primary Vehicle Identity).
 *
 * Design Rule: "False positives are more damaging to usefulness than occasional missed detections."
 *  - Tier 1 (exact normalized plate) is handled upstream at 100% confidence.
 *  - Tier 2A (near-plate 1-character OCR slip on a >=6-char plate): 82%..90% confidence (unless
 *    observed body style explicitly contradicts stored body style).
 *  - Tier 2B (3–5 char partial plate fragment from tinted/dirty/angled plate + >=1 corroborating
 *    appearance attribute: Make, BodyStyle, or Color): 72%..88% confidence.
 *  - Tier 3 (unreadable/missing plate where stored record was flagged `NO_PLATE` or without a plate,
 *    requiring ALL THREE of `Color + Make + BodyStyle` to match): 66% suspected match.
 */
object PartialMatchEngine {

    const val MIN_SUSPECTED_ALERT_CONFIDENCE = 0.65f

    fun findBestSuspectedMatch(
        candidatePlateOrFragment: String?,
        liveAppearance: StructuredAppearance?,
        incidentRows: List<DriverLog>
    ): VehicleReputation? {
        if (incidentRows.isEmpty()) return null

        val normQuery = candidatePlateOrFragment?.let { normalizePlate(it) }?.takeIf { it.length >= 3 }
        val live = liveAppearance ?: StructuredAppearance()
        if (normQuery == null && live.isEmpty) return null

        // Group known plates first
        val byPlate = incidentRows.filter { !it.plateNormalized.isNullOrBlank() }
            .groupBy { it.plateNormalized!! }

        var bestRep: VehicleReputation? = null
        var bestConf = 0f

        for ((storedNorm, groupRows) in byPlate) {
            if (normQuery != null && normQuery == storedNorm) continue // Tier 1 handles exact matches

            val storedAppearance = resolveStoredAppearance(groupRows)
            val attrAgreement = scoreAppearanceAgreement(live, storedAppearance)

            var candidateConf = 0f

            if (normQuery != null) {
                // Tier 2A: 1-character edit distance on full-length plates (>=6 chars)
                if (normQuery.length >= 6 && storedNorm.length >= 6 &&
                    editDistanceAtMostOne(normQuery, storedNorm)
                ) {
                    if (!attrAgreement.bodyConflict) {
                        candidateConf = (0.82f + attrAgreement.matchCount * 0.04f).coerceAtMost(0.90f)
                    }
                }
                // Tier 2B: 3-5 char partial substring fragment + appearance corroboration
                else if (normQuery.length in 3..5 && storedNorm.contains(normQuery)) {
                    val hasPlausibleMix = normQuery.any { it.isDigit() } &&
                        (normQuery.any { it.isLetter() } || normQuery.length >= 4)
                    if (hasPlausibleMix && attrAgreement.matchCount >= 1 && !attrAgreement.bodyConflict) {
                        val base = if (normQuery.length >= 4) 0.74f else 0.68f
                        candidateConf = (base + attrAgreement.matchCount * 0.05f).coerceAtMost(0.88f)
                    }
                }
            }

            if (candidateConf >= MIN_SUSPECTED_ALERT_CONFIDENCE && candidateConf > bestConf) {
                val rep = computeVehicleReputation(groupRows, matchConfidence = candidateConf)
                if (rep != null && !rep.isPureAngel) {
                    bestConf = candidateConf
                    bestRep = rep
                }
            }
        }

        if (bestRep != null) return bestRep

        // Tier 3: Obscured / missing plate appearance match (only when no plate read at all, and all 3
        // attributes Color + Make + BodyStyle match a BAD record flagged NO_PLATE or plate-less).
        if (normQuery == null &&
            !live.color.isNullOrBlank() &&
            !live.make.isNullOrBlank() &&
            !live.bodyStyle.isNullOrBlank()
        ) {
            val platelessOrObscured = incidentRows.filter { row ->
                row.rating == "BAD" && (
                    row.plateNormalized.isNullOrBlank() ||
                        BehaviorTag.NO_PLATE in BehaviorTag.parseList(row.behaviorTags)
                    )
            }
            for (row in platelessOrObscured) {
                val stored = resolveStoredAppearance(listOf(row))
                if (stored.color.equals(live.color, ignoreCase = true) &&
                    stored.make.equals(live.make, ignoreCase = true) &&
                    stored.bodyStyle.equals(live.bodyStyle, ignoreCase = true)
                ) {
                    val label = row.plateOcr ?: listOfNotNull(stored.color, stored.make, stored.bodyStyle).joinToString(" ")
                    val synthetic = row.copy(
                        plateOcr = label,
                        plateNormalized = "APPEARANCE:${stored.color}_${stored.make}_${stored.bodyStyle}".uppercase()
                    )
                    val rep = computeVehicleReputation(listOf(synthetic), matchConfidence = 0.66f)
                    if (rep != null) return rep
                }
            }
        }

        return null
    }

    private data class AgreementResult(val matchCount: Int, val bodyConflict: Boolean)

    private fun scoreAppearanceAgreement(
        live: StructuredAppearance,
        stored: StructuredAppearance
    ): AgreementResult {
        var matches = 0
        var bodyConflict = false

        if (!live.make.isNullOrBlank() && !stored.make.isNullOrBlank() &&
            live.make.equals(stored.make, ignoreCase = true)
        ) {
            matches++
        }
        if (!live.bodyStyle.isNullOrBlank() && !stored.bodyStyle.isNullOrBlank()) {
            if (live.bodyStyle.equals(stored.bodyStyle, ignoreCase = true)) {
                matches++
            } else {
                bodyConflict = true
            }
        }
        if (!live.color.isNullOrBlank() && !stored.color.isNullOrBlank() &&
            live.color.equals(stored.color, ignoreCase = true)
        ) {
            matches++
        }
        return AgreementResult(matches, bodyConflict)
    }

    internal fun resolveStoredAppearance(rows: List<DriverLog>): StructuredAppearance {
        val color = rows.firstNotNullOfOrNull { it.vehicleColor?.takeIf { s -> s.isNotBlank() } }
        val make = rows.firstNotNullOfOrNull { it.vehicleMake?.takeIf { s -> s.isNotBlank() } }
        val body = rows.firstNotNullOfOrNull { it.vehicleBodyStyle?.takeIf { s -> s.isNotBlank() } }
        if (color != null || make != null || body != null) {
            val fallback = VehicleAppearance.extractFromFreeText(
                rows.firstNotNullOfOrNull { it.vehicleMmc },
                rows.firstNotNullOfOrNull { it.notes }
            )
            return StructuredAppearance(
                color = color ?: fallback.color,
                make = make ?: fallback.make,
                bodyStyle = body ?: fallback.bodyStyle
            )
        }
        return VehicleAppearance.extractFromFreeText(
            rows.firstNotNullOfOrNull { it.vehicleMmc },
            rows.firstNotNullOfOrNull { it.notes }
        )
    }

    /** Fast O(N) check whether two strings have Levenshtein edit distance <= 1. */
    internal fun editDistanceAtMostOne(a: String, b: String): Boolean {
        val la = a.length
        val lb = b.length
        if (abs(la - lb) > 1) return false
        if (la == lb) {
            var diffs = 0
            for (i in 0 until la) {
                if (a[i] != b[i] && ++diffs > 1) return false
            }
            return diffs <= 1
        }
        // Length differs by 1: check single insertion/deletion
        val shorter = if (la < lb) a else b
        val longer = if (la < lb) b else a
        var i = 0
        var j = 0
        var skipped = false
        while (i < shorter.length && j < longer.length) {
            if (shorter[i] == longer[j]) {
                i++
                j++
            } else {
                if (skipped) return false
                skipped = true
                j++
            }
        }
        return true
    }
}
