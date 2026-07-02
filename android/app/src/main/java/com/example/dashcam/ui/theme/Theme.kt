package com.ct3d.jolt.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

/**
 * Single dark color scheme for Jolt. The app is dark-theme only (vehicle-mounted, daytime use),
 * so there is no light scheme. Semantic extras that don't fit a Material role live in [JoltColors].
 */
private val JoltDarkColorScheme = darkColorScheme(
    primary      = JoltRed,
    onPrimary    = Color.White,
    secondary    = JoltGreen,
    onSecondary  = Color.Black,
    tertiary     = JoltInfoBlue,
    onTertiary   = Color.Black,
    background   = JoltBackground,
    onBackground = Color.White,
    surface      = JoltSurface,
    onSurface    = Color.White,
)

@Composable
fun JoltTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = JoltDarkColorScheme,
        content = content
    )
}
