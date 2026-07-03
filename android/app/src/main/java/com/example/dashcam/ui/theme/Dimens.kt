package com.ct3d.jolt.ui.theme

import androidx.compose.ui.unit.dp

/**
 * 4dp-based spacing/dimension scale. Prefer these named steps over ad-hoc dp values so gaps and
 * padding stay on a consistent rhythm across screens.
 */
object Dimens {
    // Spacing scale (4 / 8 / 12 / 16 / 24)
    val spaceXs = 4.dp
    val spaceSm = 8.dp
    val spaceMd = 12.dp
    val spaceLg = 16.dp
    val spaceXl = 24.dp

    // Component-specific
    val alertBorder = 5.dp   // known-bad-driver pulsing border (reduced from 10dp in A13)
    val emptyIcon   = 48.dp  // empty-state icon size
}
