package com.ct3d.jolt

import com.ct3d.jolt.camera.TelephotoAnalyzer
import com.ct3d.jolt.data.BehaviorTag
import com.ct3d.jolt.data.DriverLog
import com.ct3d.jolt.data.JoltSeverity
import com.ct3d.jolt.data.computeVehicleReputation
import com.ct3d.jolt.data.normalizePlate
import com.ct3d.jolt.ml.YoloV8Detector
import com.ct3d.jolt.service.FieldTelemetryLogger
import com.ct3d.jolt.speech.VoiceCommand
import com.ct3d.jolt.speech.VoiceCommandParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class JoltCoreLogicTest {

    @Test
    fun normalizePlate_foldsCommonOcrConfusionsAndStripsSymbols() {
        assertEquals("8X145220", normalizePlate("bxi-4522o"))
        assertEquals("LT2379", normalizePlate(" lt 2379 "))
        assertEquals("5018", normalizePlate("SOIB"))
    }

    @Test
    fun extractPlateToken_ignoresStateHeaderAndFindsStandardToken() {
        val raw = "CONNECTICUT LT2379 CONSTITUTION STATE"
        assertEquals("LT2379", TelephotoAnalyzer.extractPlateToken(raw))
    }

    @Test
    fun extractPlateToken_handlesSpaceSplitPlateTokens() {
        assertEquals("ABC1234", TelephotoAnalyzer.extractPlateToken("NEW YORK ABC 1234 EMPIRE STATE"))
        assertEquals("123ABC", TelephotoAnalyzer.extractPlateToken("123 ABC"))
    }

    @Test
    fun extractPlateToken_rejectsPureStateWordsAndNoise() {
        assertNull(TelephotoAnalyzer.extractPlateToken("CONNECTICUT CONSTITUTION STATE"))
        assertNull(TelephotoAnalyzer.extractPlateToken("TEXAS LONE STAR"))
        assertNull(TelephotoAnalyzer.extractPlateToken("123456"))
    }

    @Test
    fun ocrVoteBuffer_requiresMinimumVotesAndSupportsMultiPlateFrames() {
        val voter = TelephotoAnalyzer.OcrVoteBuffer(windowSize = 5, minVotes = 2)

        // Frame 1: two plates seen once -> neither confirmed yet
        val f1 = voter.registerFrame(listOf("LT2379", "AB1234"))
        assertNull(f1.primaryConsensus)
        assertTrue(f1.confirmedTokens.isEmpty())

        // Frame 2: noisy misread on primary, second plate repeats -> AB1234 confirmed!
        val f2 = voter.registerFrame(listOf("LT2879", "AB1234"))
        assertEquals("AB1234", f2.primaryConsensus)
        assertEquals(listOf("AB1234"), f2.confirmedTokens)

        // Frame 3: both genuine plates repeat -> both confirmed, AB1234 has 3 votes vs LT2379's 2
        val f3 = voter.registerFrame(listOf("LT2379", "AB1234"))
        assertEquals("AB1234", f3.primaryConsensus)
        assertEquals(listOf("AB1234", "LT2379"), f3.confirmedTokens)

        // Reset clears all votes
        voter.reset()
        val afterReset = voter.registerFrame(listOf("LT2379"))
        assertNull(afterReset.primaryConsensus)
    }

    @Test
    fun ocrVoteBuffer_breaksTiesUsingPlateFormatPlausibility() {
        val voter = TelephotoAnalyzer.OcrVoteBuffer(windowSize = 5, minVotes = 2)
        voter.registerFrame(listOf("1234567X"))
        voter.registerFrame(listOf("1234567X"))
        voter.registerFrame(listOf("AB1234"))
        val res = voter.registerFrame(listOf("XYZ999", "AB1234"))
        // Both "1234567X" and "AB1234" have 2 votes and neither is first token of frame 4;
        // plateFormatScore prefers "AB1234" (score 1) over "1234567X" (score 0).
        assertEquals("AB1234", res.primaryConsensus)
    }

    @Test
    fun yoloParseOutput_autoDetectsNormalizedAndPixelCoordsAndAppliesNms() {
        // Normalized coords [0..1]
        val normOutput = Array(5) { FloatArray(3) }
        // Anchor 0: high-conf box centered at (0.5, 0.5), size (0.2, 0.1)
        normOutput[0][0] = 0.5f; normOutput[1][0] = 0.5f; normOutput[2][0] = 0.2f; normOutput[3][0] = 0.1f; normOutput[4][0] = 0.90f
        // Anchor 1: overlapping lower-conf duplicate
        normOutput[0][1] = 0.51f; normOutput[1][1] = 0.50f; normOutput[2][1] = 0.2f; normOutput[3][1] = 0.1f; normOutput[4][1] = 0.60f
        // Anchor 2: below threshold
        normOutput[0][2] = 0.2f; normOutput[1][2] = 0.2f; normOutput[2][2] = 0.1f; normOutput[3][2] = 0.1f; normOutput[4][2] = 0.10f

        val parsedNorm = YoloV8Detector.parseOutputBoxes(normOutput, 1000, 2000, 0.25f)
        assertEquals(2, parsedNorm.size)
        val nmsNorm = YoloV8Detector.applyNmsBoxes(parsedNorm, 0.5f)
        assertEquals(1, nmsNorm.size)
        assertEquals(400f, nmsNorm[0].left, 0.01f)
        assertEquals(600f, nmsNorm[0].right, 0.01f)
        assertEquals(900f, nmsNorm[0].top, 0.01f)
        assertEquals(1100f, nmsNorm[0].bottom, 0.01f)

        // Pixel-space coords [0..640]
        val pxOutput = Array(5) { FloatArray(1) }
        pxOutput[0][0] = 320f; pxOutput[1][0] = 320f; pxOutput[2][0] = 64f; pxOutput[3][0] = 32f; pxOutput[4][0] = 0.85f
        val parsedPx = YoloV8Detector.parseOutputBoxes(pxOutput, 1280, 640, 0.25f)
        assertEquals(1, parsedPx.size)
        assertEquals(576f, parsedPx[0].left, 0.01f)
        assertEquals(704f, parsedPx[0].right, 0.01f)
        assertEquals(304f, parsedPx[0].top, 0.01f)
        assertEquals(336f, parsedPx[0].bottom, 0.01f)
    }

    @Test
    fun csvField_quotesAndEscapesRfc4180() {
        assertEquals("", csvField(null))
        assertEquals("\"LT2379\"", csvField("LT2379"))
        assertEquals("\"Honda \"\"Civic\"\", White\"", csvField("Honda \"Civic\", White"))
    }

    @Test
    fun fieldTelemetrySummary_computesLatencyThermalMemoryAnd3FpsPass() {
        val sampleLines = buildList {
            add("SESSION\t2026-11-25T09:00:00.000-0500\tevent=USER_START_LOGGING\tdevice=Google Pixel 10\tintervalMs=500\tspeedMph=0.0")
            add("HEALTH\t2026-11-25T09:00:00.010-0500\tthermal=NONE(0)\theadroom=0.410\tbatPct=92\tbatTempC=29.5\tcharging=false\tjavaMb=42.0/512\tnativeMb=88.0\tavailRamMb=6400\tlowMem=false\tspeedMph=0.0")
            for (i in 1..25) {
                val consensus = if (i % 2 == 0) "LT2379" else "-"
                val trig = if (i % 5 == 0) "OCR_FAIL_ANCHORED" else "NONE"
                val mmcPub = if (i % 5 == 0) "Honda Accord Sedan 2012" else "-"
                val mmcConf = if (i % 5 == 0) "0.680" else "0.000"
                add(
                    "FRAME\t2026-11-25T09:00:${"%02d".format(i)}.000-0500\tdtMs=501\tdrops=0\ttotalMs=145\tyoloMs=52\tocrMs=60\tmmcMs=${if (i % 5 == 0) 33 else 0}\tgpu=1/1\tboxes=1\tboxConf=0.880\tocrTokens=${if (i % 5 == 0) "-" else "LT2379"}\tconsensus=$consensus\tmmcTrig=$trig\tmmcPub=$mmcPub\tmmcConf=$mmcConf\tmmcTop3=Honda=0.680\tspeedMph=45.5"
                )
            }
            add("HEALTH\t2026-11-25T09:30:00.000-0500\tthermal=LIGHT(1)\theadroom=0.680\tbatPct=84\tbatTempC=34.2\tcharging=false\tjavaMb=46.5/512\tnativeMb=91.5\tavailRamMb=6300\tlowMem=false\tspeedMph=52.0")
        }

        val summary = FieldTelemetryLogger.buildSummaryHeader(sampleLines)
        assertTrue(summary.contains("Analyzed frames: 25  |  Busy drops: 0"))
        assertTrue(summary.contains("Peak thermal:    LIGHT(1)"))
        assertTrue(summary.contains("Battery temp:    29.5°C .. 34.2°C  |  Battery level: 92% -> 84%"))
        assertTrue(summary.contains("Native heap MB:  start=88.0  end=91.5  peak=91.5  (delta=+3.5 MB)"))
        assertTrue(summary.contains("MMC triggered:   5 frames (anchored=5, center=0)  |  Published (>=0.25): 5"))
        assertTrue(summary.contains("PASS — p95=145ms (<290ms), thermal<=LIGHT, native heap stable."))
    }

    @Test
    fun voiceCommandParser_handlesWakeWordTagsAngelFollowUpAndCorrections() {
        // Ignore speech without wake word in normal mode
        assertNull(VoiceCommandParser.parse("weaving and speeding", awaitingFollowUpReason = false))
        assertNull(VoiceCommandParser.parse("[unk]", awaitingFollowUpReason = true))

        // One-shot wake command with descriptors + 2 behavior tags
        val cmd1 = VoiceCommandParser.parse(
            "hey jolt flag white honda suv cutting and asshole",
            awaitingFollowUpReason = false
        ) as? VoiceCommand.FlagVehicle
        assertNotNull(cmd1)
        assertEquals(listOf(BehaviorTag.CUTTING, BehaviorTag.ASSHOLE), cmd1!!.tags)
        assertEquals("white honda suv", cmd1.descriptorNotes)
        // "white" should not override severity; default severity of CUTTING/ASSHOLE is RED
        assertNull(cmd1.severityOverride)
        assertEquals(JoltSeverity.RED, BehaviorTag.defaultSeverityFor(cmd1.tags, cmd1.severityOverride))

        // Follow-up reason without wake word when awaiting reason
        val followUp = VoiceCommandParser.parse(
            "unsteady throttle and weaving",
            awaitingFollowUpReason = true
        ) as? VoiceCommand.FollowUpReason
        assertNotNull(followUp)
        assertEquals(listOf(BehaviorTag.THROTTLE, BehaviorTag.WEAVING), followUp!!.tags)

        // Angel command
        val angel = VoiceCommandParser.parse("jolt angel that silver sedan", awaitingFollowUpReason = false)
        assertTrue(angel is VoiceCommand.AngelVehicle)
        assertEquals("silver sedan", (angel as VoiceCommand.AngelVehicle).descriptorNotes)

        // Standalone wake word ("Jolt") opens awake window!
        assertEquals(
            VoiceCommand.WakeDetected,
            VoiceCommandParser.parse("jolt", awaitingFollowUpReason = false, isAwakeForCommand = false)
        )

        // Subsequent action word without repeating "Jolt" while in 8s awake window fires immediately!
        val awakeCmd = VoiceCommandParser.parse(
            "weaving",
            awaitingFollowUpReason = false,
            isAwakeForCommand = true
        ) as? VoiceCommand.FlagVehicle
        assertNotNull(awakeCmd)
        assertEquals(listOf(BehaviorTag.WEAVING), awakeCmd!!.tags)

        // Wrong car / cancel
        assertEquals(VoiceCommand.UndoOrWrongCar, VoiceCommandParser.parse("jolt wrong car", awaitingFollowUpReason = false))
        assertEquals(VoiceCommand.UndoOrWrongCar, VoiceCommandParser.parse("cancel", awaitingFollowUpReason = true))
        assertEquals(VoiceCommand.MuteAlerts, VoiceCommandParser.parse("jolt mute alerts", awaitingFollowUpReason = false))
        assertEquals(VoiceCommand.UnmuteAlerts, VoiceCommandParser.parse("jolt alerts on", awaitingFollowUpReason = false))
    }

    @Test
    fun vehicleReputation_escalatesRepeatRedToMaroonAndPreservesAngelHalo() {
        val logs = listOf(
            DriverLog(
                id = 1, rating = "BAD", plateOcr = "LT2379", plateNormalized = "LT2379",
                vehicleMmc = "Honda CR-V SUV 2012", timestamp = 1_000L, latitude = 41.7, longitude = -72.6,
                behaviorTags = "CUTTING", severity = "RED"
            ),
            DriverLog(
                id = 2, rating = "ANGEL", plateOcr = "LT2379", plateNormalized = "LT2379",
                vehicleMmc = "Honda CR-V SUV 2012", timestamp = 2_000L, latitude = 41.7, longitude = -72.6,
                severity = "ANGEL"
            ),
            DriverLog(
                id = 3, rating = "BAD", plateOcr = "LT2379", plateNormalized = "LT2379",
                vehicleMmc = "Honda CR-V SUV 2012", timestamp = 3_000L, latitude = 41.7, longitude = -72.6,
                behaviorTags = "CUTTING,TAILGATING", severity = "RED"
            )
        )

        val rep = computeVehicleReputation(logs)
        assertNotNull(rep)
        // 2 RED incidents escalate to MAROON automatically (Requirements §6)
        assertEquals(JoltSeverity.MAROON, rep!!.effectiveSeverity)
        assertEquals(2, rep.badIncidentCount)
        assertEquals(1, rep.angelCount)
        assertTrue(rep.hasAngelHalo)
        assertFalse(rep.isPureAngel)
        assertEquals("LINE CUTTER + TAILGATER", rep.primaryHudBehavior)
        assertEquals("Maroon. Repeat line cutter. 2 prior flags. Also marked angel.", rep.buildSpokenAlert())
    }
}
