# Add project specific ProGuard rules here.

# TensorFlow Lite / GPU delegate JNI bindings
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**

# ML Kit bundled text recognition
-keep class com.google.mlkit.** { *; }
-dontwarn com.google.mlkit.**

# Room entities / generated DAOs
-keep class com.ct3d.jolt.data.** { *; }
-keepclassmembers class * extends androidx.room.RoomDatabase { *; }

# OSMDroid tile providers & reflection
-keep class org.osmdroid.** { *; }
-dontwarn org.osmdroid.**

# Vosk & JNA native bindings for offline speech recognition
-keep class com.sun.jna.** { *; }
-keep class * implements com.sun.jna.** { *; }
-keep class org.vosk.** { *; }
-dontwarn com.sun.jna.**
-dontwarn org.vosk.**
