package com.ct3d.jolt.ui.theme

import androidx.compose.ui.graphics.Color

/**
 * Jolt color tokens. Every color used in the UI comes from here (or from Material named colors
 * like Color.White/Gray/Black) — no inline hex literals in composables (A12).
 */

// Core brand palette
val JoltRed        = Color(0xFFFF1744) // primary   — FLAG button, alerts, BAD rating
val JoltGreen      = Color(0xFF00E676) // secondary — active detections, bbox, ANGEL rating
val JoltInfoBlue   = Color(0xFF00B0FF) // tertiary  — info banner, MMC fallback, OCR label
val JoltBackground = Color(0xFF0A0A0A) // app background
val JoltSurface    = Color(0xFF1E1E1E) // cards & elevated surfaces

/** Semantic colors that don't map onto a standard Material color role. */
object JoltColors {
    val alertYellow     = Color(0xFFFFD700) // yellow severity / sighting pin
    val severityMaroon  = Color(0xFFAD1457) // maroon — persistent/extreme threat pattern
    val severityRed     = JoltRed           // red — immediate road risk
    val severityOrange  = Color(0xFFFF9100) // orange — meaningful caution
    val severityYellow  = alertYellow       // yellow — nuisance / behavioral tell
    val angelGreen      = JoltGreen         // green — Angel courteous/goodwill mark
    val bboxGreen       = JoltGreen         // plate bounding-box stroke
    val actionBlue      = Color(0xFF1565C0) // secondary action buttons
    val navBar          = Color(0xFF111111) // bottom navigation container
    val navIndicator    = Color(0xFF2A0A0F) // selected nav item pill (deep red)
    val cardElevated    = Color(0xFF262626) // nested stat boxes / spotlight cards
    val barTrack        = Color(0xFF303030) // analytics progress bar background track
    val watchlistAccent = Color(0xFFFFAB40) // Central CT watchlist highlight (Honda/Acura/Infiniti)
    val peerBadge       = Color(0xFF7C4DFF) // Trusted-Group peer shared badge
}
