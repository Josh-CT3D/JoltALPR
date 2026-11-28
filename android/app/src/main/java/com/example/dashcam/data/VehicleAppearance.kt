package com.ct3d.jolt.data

import kotlin.math.max
import kotlin.math.min

/**
 * Phase 3 Structured Vehicle Appearance Engine (`Color + Make + BodyStyle`).
 *
 * Solves the 2012 Stanford Cars trim-year mismatch discovered on road tests:
 *  1. Maps all 196 Stanford Cars class labels into timeless `(Make, BodyStyle)` pairs so softmax
 *     probabilities can be marginalized across trims (e.g. summing all SUV classes or all Honda
 *     classes instead of requiring a single 2012 trim to exceed 0.25).
 *  2. Extracts dominant vehicle body paint color (`White`, `Black`, `Silver`, `Gray`, `Red`,
 *     `Blue`, `Green`, `Tan`) directly from ARGB pixels in pure Kotlin (zero JNI dependency).
 *  3. Parses spoken descriptors (`"white honda suv"`, `"jeep"`) and legacy `vehicleMmc` strings
 *     into canonical structured fields for Analytics and Tier 2/3 Partial Matching.
 */
data class StructuredAppearance(
    val color: String? = null,
    val make: String? = null,
    val bodyStyle: String? = null,
    val rawTrimHint: String? = null,
    val makeConfidence: Float = 0f,
    val bodyConfidence: Float = 0f,
    val colorConfidence: Float = 0f
) {
    /** Clean, year-free vehicle summary for HUD & DB (e.g. "White Honda SUV", "Silver Sedan"). */
    fun formatSummary(): String? {
        val parts = listOfNotNull(color, make, bodyStyle).filter { it.isNotBlank() }
        return if (parts.isNotEmpty()) parts.joinToString(" ") else null
    }

    val isEmpty: Boolean
        get() = color.isNullOrBlank() && make.isNullOrBlank() && bodyStyle.isNullOrBlank()
}

object VehicleAppearance {

    val WATCHLIST_MAKES = setOf("Honda", "Acura", "Infiniti")

    private val multiWordMakes = listOf(
        "AM General" to "Hummer",
        "Aston Martin" to "Aston Martin",
        "Land Rover" to "Land Rover",
        "Mercedes-Benz" to "Mercedes-Benz",
        "Rolls-Royce" to "Rolls-Royce"
    )

    private val makeAliases = mapOf(
        "chevy" to "Chevrolet",
        "chevrolet" to "Chevrolet",
        "honda" to "Honda",
        "acura" to "Acura",
        "infiniti" to "Infiniti",
        "toyota" to "Toyota",
        "scion" to "Toyota",
        "ford" to "Ford",
        "subaru" to "Subaru",
        "bmw" to "BMW",
        "tesla" to "Tesla",
        "nissan" to "Nissan",
        "jeep" to "Jeep",
        "audi" to "Audi",
        "mercedes" to "Mercedes-Benz",
        "mercedes-benz" to "Mercedes-Benz",
        "hyundai" to "Hyundai",
        "kia" to "Kia",
        "mazda" to "Mazda",
        "volkswagen" to "Volkswagen",
        "vw" to "Volkswagen",
        "volvo" to "Volvo",
        "lexus" to "Lexus",
        "dodge" to "Dodge",
        "ram" to "Ram",
        "gmc" to "GMC",
        "cadillac" to "Cadillac",
        "buick" to "Buick",
        "chrysler" to "Chrysler",
        "lincoln" to "Lincoln",
        "porsche" to "Porsche",
        "hummer" to "Hummer",
        "mini" to "MINI",
        "fiat" to "FIAT",
        "mitsubishi" to "Mitsubishi",
        "suzuki" to "Suzuki",
        "jaguar" to "Jaguar",
        "smart" to "Smart"
    )

    private val colorAliases = mapOf(
        "white" to "White",
        "black" to "Black",
        "silver" to "Silver",
        "gray" to "Gray",
        "grey" to "Gray",
        "red" to "Red",
        "blue" to "Blue",
        "green" to "Green",
        "tan" to "Tan",
        "gold" to "Tan"
    )

    private val bodyAliases = mapOf(
        "suv" to "SUV",
        "crossover" to "SUV",
        "sedan" to "Sedan",
        "truck" to "Pickup",
        "pickup" to "Pickup",
        "van" to "Van",
        "minivan" to "Van",
        "coupe" to "Coupe",
        "wagon" to "Wagon/Hatch",
        "hatchback" to "Wagon/Hatch",
        "hatch" to "Wagon/Hatch",
        "convertible" to "Convertible"
    )

    /**
     * Maps any Stanford Cars class name (underscore or space separated) to `(Make, BodyStyle)`.
     */
    fun parseStanfordCarsClass(rawClass: String): Pair<String, String> {
        val clean = rawClass.replace('_', ' ').trim()
        var make: String? = null
        for ((prefix, canonical) in multiWordMakes) {
            if (clean.startsWith(prefix, ignoreCase = true)) {
                make = canonical
                break
            }
        }
        if (make == null) {
            val firstWord = clean.substringBefore(' ')
            make = makeAliases[firstWord.lowercase()] ?: firstWord.replaceFirstChar { it.uppercase() }
        }

        val padded = " ${clean.lowercase()} "
        val bodyStyle = when {
            padded.contains(" crew cab ") || padded.contains(" regular cab ") ||
                padded.contains(" extended cab ") || padded.contains(" supercab ") ||
                padded.contains(" club cab ") || padded.contains(" quad cab ") ||
                padded.contains(" pickup ") -> "Pickup"
            padded.contains(" suv ") || padded.contains(" trailblazer ") ||
                padded.contains(" ascender ") || padded.contains(" rainier ") ||
                padded.contains(" wrangler ") -> "SUV"
            padded.contains(" minivan ") || padded.contains(" cargo van ") ||
                padded.contains(" passenger van ") || padded.contains(" sprinter ") ||
                padded.contains(" van ") -> "Van"
            padded.contains(" convertible ") || padded.contains(" roadster ") -> "Convertible"
            padded.contains(" wagon ") || padded.contains(" hatchback ") ||
                padded.contains(" abarth ") || padded.contains(" hhr ") ||
                padded.contains(" veloster ") || padded.contains(" zdx ") -> "Wagon/Hatch"
            padded.contains(" coupe ") || padded.contains(" challenger ") ||
                padded.contains(" cobalt ss ") || padded.contains(" integra ") ||
                padded.contains(" xk ") -> "Coupe"
            else -> "Sedan"
        }
        return make to bodyStyle
    }

    /**
     * Extracts structured `(color, make, bodyStyle)` from combined spoken notes and/or `vehicleMmc`
     * strings (handles both modern `"White Honda SUV"` and legacy `"Honda Accord Coupe 2012"`).
     */
    fun extractFromFreeText(primaryText: String?, secondaryNotes: String? = null): StructuredAppearance {
        val combined = listOfNotNull(secondaryNotes, primaryText)
            .joinToString(" ")
            .replace('_', ' ')
            .trim()
        if (combined.isEmpty()) return StructuredAppearance()

        val tokens = combined.lowercase()
            .replace(Regex("[^a-z0-9\\- ]"), " ")
            .split(Regex("\\s+"))
            .filter { it.isNotEmpty() }

        var color: String? = null
        var make: String? = null
        var body: String? = null

        // Check multi-word makes first
        for ((prefix, canonical) in multiWordMakes) {
            if (combined.contains(prefix, ignoreCase = true)) {
                make = canonical
                break
            }
        }

        for (tok in tokens) {
            if (color == null) color = colorAliases[tok]
            if (make == null) make = makeAliases[tok]
            if (body == null) body = bodyAliases[tok]
        }

        // Check Stanford-Cars cab/body patterns if body wasn't an explicit single word
        if (body == null && primaryText != null && primaryText.any { it.isDigit() }) {
            val (_, parsedBody) = parseStanfordCarsClass(primaryText)
            body = parsedBody
        }

        return StructuredAppearance(
            color = color,
            make = make,
            bodyStyle = body,
            rawTrimHint = primaryText
        )
    }

    /**
     * Marginalizes a 196-way Stanford Cars softmax probability vector into `(Make, BodyStyle)`
     * distributions and fuses with pixel-estimated [paintColor].
     *
     * Avoids misleading 1991-2012 exotic/vintage labels (`Audi 100 Wagon 1994`, `Lamborghini`,
     * `Geo Metro`) unless a mainstream modern make has genuine summed probability support!
     */
    fun fuseMmcProbabilitiesAndColor(
        labels: List<String>,
        probs: FloatArray,
        paintColor: Pair<String?, Float>
    ): StructuredAppearance {
        val makeProbs = HashMap<String, Float>(48)
        val bodyProbs = HashMap<String, Float>(8)

        val n = min(labels.size, probs.size)
        var bestIdx = 0
        for (i in 0 until n) {
            val p = probs[i]
            if (p > probs[bestIdx]) bestIdx = i
            val (mk, bd) = parseStanfordCarsClass(labels[i])
            makeProbs[mk] = (makeProbs[mk] ?: 0f) + p
            bodyProbs[bd] = (bodyProbs[bd] ?: 0f) + p
        }

        // Collapse rare/exotic Stanford Cars makes that never represent normal commuter traffic unless ultra-confident
        val exoticMakes = setOf(
            "Ferrari", "Lamborghini", "Bugatti", "McLaren", "Spyker", "Maybach",
            "Rolls-Royce", "Bentley", "Aston Martin", "Fisker", "Geo", "Eagle", "Daewoo", "Plymouth", "Isuzu"
        )

        val bestBodyEntry = bodyProbs.maxByOrNull { it.value }
        val chosenBody = bestBodyEntry?.takeIf { it.value >= MIN_MARGINAL_BODY_PROB }?.key
        val bodyConf = bestBodyEntry?.value ?: 0f

        val bestMakeEntry = makeProbs.entries
            .filter { (mk, p) -> if (mk in exoticMakes) p >= 0.70f else p >= MIN_MARGINAL_MAKE_PROB }
            .maxByOrNull { it.value }
        val chosenMake = bestMakeEntry?.key
        val makeConf = bestMakeEntry?.value ?: 0f

        val (colorName, colorConf) = paintColor
        val chosenColor = colorName?.takeIf { colorConf >= MIN_COLOR_CONFIDENCE }

        val topRaw = if (n > 0) labels[bestIdx].replace('_', ' ') else null

        return StructuredAppearance(
            color = chosenColor,
            make = chosenMake,
            bodyStyle = chosenBody,
            rawTrimHint = topRaw,
            makeConfidence = makeConf,
            bodyConfidence = bodyConf,
            colorConfidence = colorConf
        )
    }

    /**
     * Pure-Kotlin HSV dominant vehicle paint estimator over a `[width x height]` ARGB pixel array.
     *
     * Samples the lower-central painted body region (`x` in `24%..76%`, `y` in `34%..72%`),
     * avoiding upper rear-windshield glass/sky and lower road asphalt.
     */
    fun estimateBodyColorFromPixels(
        argbPixels: IntArray,
        width: Int,
        height: Int
    ): Pair<String?, Float> {
        if (width <= 4 || height <= 4 || argbPixels.size < width * height) return null to 0f

        val x0 = (width * 0.24f).toInt().coerceIn(0, width - 1)
        val x1 = (width * 0.76f).toInt().coerceIn(x0 + 1, width)
        val y0 = (height * 0.34f).toInt().coerceIn(0, height - 1)
        val y1 = (height * 0.72f).toInt().coerceIn(y0 + 1, height)

        val buckets = LinkedHashMap<String, Int>().apply {
            put("White", 0)
            put("Black", 0)
            put("Silver", 0)
            put("Gray", 0)
            put("Red", 0)
            put("Blue", 0)
            put("Green", 0)
            put("Tan", 0)
        }

        var validSamples = 0
        val step = if (width >= 128) 2 else 1

        var y = y0
        while (y < y1) {
            val rowOffset = y * width
            var x = x0
            while (x < x1) {
                val px = argbPixels[rowOffset + x]
                val r = ((px shr 16) and 0xFF) / 255f
                val g = ((px shr 8) and 0xFF) / 255f
                val b = (px and 0xFF) / 255f

                val cMax = max(r, max(g, b))
                val cMin = min(r, min(g, b))
                val delta = cMax - cMin
                val sat = if (cMax <= 0.001f) 0f else delta / cMax
                val value = cMax

                // Skip blown specular sun highlights
                if (value > 0.985f && sat < 0.03f) {
                    x += step
                    continue
                }

                validSamples++
                val bucket = if (sat < 0.17f || (value < 0.18f && sat < 0.32f)) {
                    when {
                        value >= 0.74f && sat < 0.13f -> "White"
                        value <= 0.21f -> "Black"
                        value >= 0.50f -> "Silver"
                        else -> "Gray"
                    }
                } else {
                    // Chromatic paint — compute Hue in degrees [0..360)
                    val rawHue = when {
                        delta < 0.0001f -> 0f
                        cMax == r -> 60f * (((g - b) / delta) % 6f)
                        cMax == g -> 60f * (((b - r) / delta) + 2f)
                        else -> 60f * (((r - g) / delta) + 4f)
                    }
                    val hue = if (rawHue < 0f) rawHue + 360f else rawHue
                    when {
                        hue < 18f || hue >= 338f -> "Red"
                        hue in 18f..52f -> if (sat < 0.45f) "Tan" else "Red"
                        hue in 52f..165f -> "Green"
                        hue in 165f..265f -> "Blue"
                        else -> "Red"
                    }
                }
                buckets[bucket] = (buckets[bucket] ?: 0) + 1
                x += step
            }
            y += step
        }

        if (validSamples < 16) return null to 0f

        // Rear windows & dark trim slightly inflate "Black"; require stronger share for Black
        val best = buckets.entries.maxByOrNull { (k, count) ->
            if (k == "Black") (count * 0.78f).toInt() else count
        } ?: return null to 0f

        val share = best.value.toFloat() / validSamples.toFloat()
        if (best.key == "Black" && share < 0.42f) {
            // Check second-best painted color (often dark Gray/Blue/Silver car with dark rear glass)
            val runnerUp = buckets.entries.filter { it.key != "Black" }.maxByOrNull { it.value }
            if (runnerUp != null && runnerUp.value.toFloat() / validSamples >= 0.28f) {
                return runnerUp.key to (runnerUp.value.toFloat() / validSamples)
            }
        }
        return if (share >= MIN_COLOR_CONFIDENCE) best.key to share else null to share
    }

    const val MIN_MARGINAL_BODY_PROB = 0.34f
    const val MIN_MARGINAL_MAKE_PROB = 0.32f
    const val MIN_COLOR_CONFIDENCE = 0.30f
}
