package com.ct3d.jolt.ui.theme

import androidx.compose.ui.graphics.Color

/**
 * Jolt color tokens. Every color used in the UI comes from here (or from Material named colors
 * like Color.White/Gray/Black) — no inline hex literals in composables.
 *
 * The core brand palette below is also wired into the Material dark color scheme in [JoltTheme],
 * so Material components (buttons, cards, nav bar) pick these up by default.
 */

// Core brand palette
val JoltRed        = Color(0xFFFF1744) // primary   — FLAG button, alerts, BAD rating
val JoltGreen      = Color(0xFF00E676) // secondary — active detections, bbox, GOOD rating
val JoltInfoBlue   = Color(0xFF00B0FF) // tertiary  — info banner, MMC fallback, OCR label
val JoltBackground = Color(0xFF0A0A0A) // app background
val JoltSurface    = Color(0xFF1E1E1E) // cards & elevated surfaces

/** Semantic colors that don't map onto a standard Material color role. */
object JoltColors {
    val alertYellow  = Color(0xFFFFD700) // known-bad-driver pulsing border + warning card
    val bboxGreen    = JoltGreen         // plate bounding-box stroke
    val actionBlue   = Color(0xFF1565C0) // secondary action buttons (re-center FAB, export CSV)
    val navBar       = Color(0xFF111111) // bottom navigation container
    val navIndicator = Color(0xFF2A0A0F) // selected nav item pill (deep red)
}
