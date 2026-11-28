package com.ct3d.jolt

import com.ct3d.jolt.camera.TelephotoAnalyzer
import com.ct3d.jolt.data.AppearanceEncounter
import com.ct3d.jolt.data.BehaviorTag
import com.ct3d.jolt.data.DriverLog
import com.ct3d.jolt.data.JoltSeverity
import com.ct3d.jolt.data.PartialMatchEngine
import com.ct3d.jolt.data.StructuredAppearance
import com.ct3d.jolt.data.TrendsAnalyticsEngine
import com.ct3d.jolt.data.TrustedGroupSync
import com.ct3d.jolt.data.VehicleAppearance
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
    fun extractPlateToken_andPartialFragment_handleFullAndObscuredPlates() {
        val raw = "CONNECTICUT LT2379 CONSTITUTION STATE"
        assertEquals("LT2379", TelephotoAnalyzer.extractPlateToken(raw))
        assertEquals("ABC1234", TelephotoAnalyzer.extractPlateToken("NEW YORK ABC 1234 EMPIRE STATE"))
        assertNull(TelephotoAnalyzer.extractPlateToken("CONNECTICUT CONSTITUTION STATE"))

        // Partial plate fragment extraction (e.g. tinted cover / partial glare)
        assertEquals("AK67", TelephotoAnalyzer.extractPartialPlateFragment("CONNECTICUT AK67 STATE"))
        assertEquals("9460", TelephotoAnalyzer.extractPartialPlateFragment("CONSTITUTION 9460"))
        assertNull(TelephotoAnalyzer.extractPartialPlateFragment("CONNECTICUT STATE"))
    }

    @Test
    fun ocrVoteBuffer_requiresMinimumVotesAndSupportsMultiPlateFrames() {
        val voter = TelephotoAnalyzer.OcrVoteBuffer(windowSize = 5, minVotes = 2)

        val f1 = voter.registerFrame(listOf("LT2379", "AB1234"))
        assertNull(f1.primaryConsensus)
        assertTrue(f1.confirmedTokens.isEmpty())

        val f2 = voter.registerFrame(listOf("LT2879", "AB1234"))
        assertEquals("AB1234", f2.primaryConsensus)
        assertEquals(listOf("AB1234"), f2.confirmedTokens)

        val f3 = voter.registerFrame(listOf("LT2379", "AB1234"))
        assertEquals("AB1234", f3.primaryConsensus)
        assertEquals(listOf("AB1234", "LT2379"), f3.confirmedTokens)

        voter.reset()
        assertNull(voter.registerFrame(listOf("LT2379")).primaryConsensus)
    }

    @Test
    fun ocrVoteBuffer_breaksTiesUsingPlateFormatPlausibility() {
        val voter = TelephotoAnalyzer.OcrVoteBuffer(windowSize = 5, minVotes = 2)
        voter.registerFrame(listOf("1234567X"))
        voter.registerFrame(listOf("1234567X"))
        voter.registerFrame(listOf("AB1234"))
        val res = voter.registerFrame(listOf("XYZ999", "AB1234"))
        assertEquals("AB1234", res.primaryConsensus)
    }

    @Test
    fun yoloParseOutput_autoDetectsNormalizedAndPixelCoordsAndAppliesNms() {
        val normOutput = Array(5) { FloatArray(3) }
        normOutput[0][0] = 0.5f; normOutput[1][0] = 0.5f; normOutput[2][0] = 0.2f; normOutput[3][0] = 0.1f; normOutput[4][0] = 0.90f
        normOutput[0][1] = 0.51f; normOutput[1][1] = 0.50f; normOutput[2][1] = 0.2f; normOutput[3][1] = 0.1f; normOutput[4][1] = 0.60f
        normOutput[0][2] = 0.2f; normOutput[1][2] = 0.2f; normOutput[2][2] = 0.1f; normOutput[3][2] = 0.1f; normOutput[4][2] = 0.10f

        val parsedNorm = YoloV8Detector.parseOutputBoxes(normOutput, 1000, 2000, 0.25f)
        assertEquals(2, parsedNorm.size)
        val nmsNorm = YoloV8Detector.applyNmsBoxes(parsedNorm, 0.5f)
        assertEquals(1, nmsNorm.size)
        assertEquals(400f, nmsNorm[0].left, 0.01f)
        assertEquals(600f, nmsNorm[0].right, 0.01f)
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
                val mmcPub = if (i % 5 == 0) "White Honda Sedan" else "-"
                val mmcConf = if (i % 5 == 0) "0.680" else "0.000"
                add(
                    "FRAME\t2026-11-25T09:00:${"%02d".format(i)}.000-0500\tdtMs=501\tdrops=0\ttotalMs=145\tyoloMs=52\tocrMs=60\tmmcMs=${if (i % 5 == 0) 33 else 0}\tgpu=1/1\tboxes=1\tboxConf=0.880\tocrTokens=${if (i % 5 == 0) "-" else "LT2379"}\tconsensus=$consensus\tmmcTrig=$trig\tmmcPub=$mmcPub\tmmcConf=$mmcConf\tmmcTop3=Honda=0.680\tspeedMph=45.5"
                )
            }
            add("HEALTH\t2026-11-25T09:30:00.000-0500\tthermal=LIGHT(1)\theadroom=0.680\tbatPct=84\tbatTempC=34.2\tcharging=false\tjavaMb=46.5/512\tnativeMb=91.5\tavailRamMb=6300\tlowMem=false\tspeedMph=52.0")
        }

        val summary = FieldTelemetryLogger.buildSummaryHeader(sampleLines)
        assertTrue(summary.contains("Analyzed frames: 25  |  Busy drops: 0"))
        assertTrue(summary.contains("PASS — p95=145ms (<290ms), thermal<=LIGHT, native heap stable."))
    }

    @Test
    fun voiceCommandParser_rejectsRoadNoiseBoltAndHandlesRepeatAlert() {
        // Phase 2.1: "bolt" and "volt" road-noise false-wakes are now ignored!
        assertNull(VoiceCommandParser.parse("hey bolt", awaitingFollowUpReason = false, isAwakeForCommand = false))
        assertNull(VoiceCommandParser.parse("volt", awaitingFollowUpReason = false, isAwakeForCommand = false))

        // Genuine "jolt" still wakes cleanly
        assertEquals(
            VoiceCommand.WakeDetected,
            VoiceCommandParser.parse("hey jolt", awaitingFollowUpReason = false, isAwakeForCommand = false)
        )

        // Repeat last alert command
        assertEquals(
            VoiceCommand.RepeatLastAlert,
            VoiceCommandParser.parse("jolt what was that", awaitingFollowUpReason = false, isAwakeForCommand = false)
        )

        val cmd1 = VoiceCommandParser.parse(
            "hey jolt flag white honda suv cutting and asshole",
            awaitingFollowUpReason = false
        ) as? VoiceCommand.FlagVehicle
        assertNotNull(cmd1)
        assertEquals(listOf(BehaviorTag.CUTTING, BehaviorTag.ASSHOLE), cmd1!!.tags)
        assertEquals("white honda suv", cmd1.descriptorNotes)
    }

    @Test
    fun vehicleReputation_escalatesRepeatRedToMaroonAndPreservesAngelHalo() {
        val logs = listOf(
            DriverLog(
                id = 1, rating = "BAD", plateOcr = "LT2379", plateNormalized = "LT2379",
                vehicleMmc = "White Honda SUV", timestamp = 1_000L, latitude = 41.7, longitude = -72.6,
                behaviorTags = "CUTTING", severity = "RED"
            ),
            DriverLog(
                id = 2, rating = "ANGEL", plateOcr = "LT2379", plateNormalized = "LT2379",
                vehicleMmc = "White Honda SUV", timestamp = 2_000L, latitude = 41.7, longitude = -72.6,
                severity = "ANGEL"
            ),
            DriverLog(
                id = 3, rating = "BAD", plateOcr = "LT2379", plateNormalized = "LT2379",
                vehicleMmc = "White Honda SUV", timestamp = 3_000L, latitude = 41.7, longitude = -72.6,
                behaviorTags = "CUTTING,TAILGATING", severity = "RED"
            )
        )

        val rep = computeVehicleReputation(logs)
        assertNotNull(rep)
        assertEquals(JoltSeverity.MAROON, rep!!.effectiveSeverity)
        assertEquals("Maroon. Repeat line cutter. 2 prior flags. Also marked angel.", rep.buildSpokenAlert())
    }

    @Test
    fun vehicleAppearance_marginalizesSoftmaxAndEstimatesBodyPaintColor() {
        assertEquals("Hummer" to "SUV", VehicleAppearance.parseStanfordCarsClass("AM_General_Hummer_SUV_2000"))
        assertEquals("Ford" to "Pickup", VehicleAppearance.parseStanfordCarsClass("Ford_F-150_Regular_Cab_2012"))
        assertEquals("Honda" to "Van", VehicleAppearance.parseStanfordCarsClass("Honda_Odyssey_Minivan_2012"))

        // Synthetic 32x32 pure white vehicle body pixel buffer
        val whitePixels = IntArray(32 * 32) { 0xFFF2F3F5.toInt() }
        val (paintColor, colorConf) = VehicleAppearance.estimateBodyColorFromPixels(whitePixels, 32, 32)
        assertEquals("White", paintColor)
        assertTrue(colorConf > 0.85f)

        // Marginalization: 3 SUV models each at 0.22 (none individually >= 0.25!) sum to Body=SUV (0.66)
        val labels = listOf(
            "Hyundai_Tucson_SUV_2012",
            "Jeep_Compass_SUV_2012",
            "GMC_Terrain_SUV_2012",
            "Audi_100_Wagon_1994"
        )
        val probs = floatArrayOf(0.22f, 0.22f, 0.22f, 0.34f)
        val fused = VehicleAppearance.fuseMmcProbabilitiesAndColor(labels, probs, paintColor to colorConf)
        assertEquals("White", fused.color)
        assertEquals("SUV", fused.bodyStyle)
        assertEquals(0.66f, fused.bodyConfidence, 0.01f)
    }

    @Test
    fun partialMatchEngine_matchesNearPlateAndCorroboratedPartialFragment() {
        val storedRows = listOf(
            DriverLog(
                id = 10, rating = "BAD", plateOcr = "AK67370", plateNormalized = "AK67370",
                vehicleMmc = "White Honda SUV", vehicleColor = "White", vehicleMake = "Honda", vehicleBodyStyle = "SUV",
                timestamp = 5_000L, latitude = 41.7, longitude = -72.6,
                behaviorTags = "WEAVING", severity = "ORANGE"
            )
        )

        // 1-char OCR slip ("AK67378" vs "AK67370") -> high-confidence suspected match
        val nearMatch = PartialMatchEngine.findBestSuspectedMatch(
            candidatePlateOrFragment = "AK67378",
            liveAppearance = StructuredAppearance(color = "White", bodyStyle = "SUV"),
            incidentRows = storedRows
        )
        assertNotNull(nearMatch)
        assertEquals("AK67370", nearMatch!!.plateDisplay)
        assertTrue(nearMatch.matchConfidence in 0.82f..0.90f)
        assertTrue(nearMatch.buildSpokenAlert().startsWith("Suspected Orange."))

        // 4-char partial fragment ("K673") WITHOUT corroborating appearance -> rejected (prevents false alarms)
        val bareFragment = PartialMatchEngine.findBestSuspectedMatch(
            candidatePlateOrFragment = "K673",
            liveAppearance = StructuredAppearance(),
            incidentRows = storedRows
        )
        assertNull(bareFragment)

        // 4-char partial fragment ("K673") WITH matching Honda + SUV -> accepted!
        val corroboratedFragment = PartialMatchEngine.findBestSuspectedMatch(
            candidatePlateOrFragment = "K673",
            liveAppearance = StructuredAppearance(make = "Honda", bodyStyle = "SUV"),
            incidentRows = storedRows
        )
        assertNotNull(corroboratedFragment)
        assertTrue(corroboratedFragment!!.matchConfidence >= 0.80f)
    }

    @Test
    fun trendsAnalyticsAndTrustedGroupBundle_computeNormalizedRatesAndStripGps() {
        val logs = listOf(
            DriverLog(
                id = 1, rating = "BAD", plateOcr = "AB1234", plateNormalized = "A81234",
                vehicleMmc = "White Honda SUV", vehicleColor = "White", vehicleMake = "Honda", vehicleBodyStyle = "SUV",
                timestamp = 1_000L, latitude = 41.7658, longitude = -72.6734,
                behaviorTags = "CUTTING", severity = "RED"
            ),
            DriverLog(
                id = 2, rating = "BAD", plateOcr = "CD5678", plateNormalized = "CD5678",
                vehicleMmc = "Black Acura Sedan", vehicleColor = "Black", vehicleMake = "Acura", vehicleBodyStyle = "Sedan",
                timestamp = 2_000L, latitude = 41.7600, longitude = -72.6700,
                behaviorTags = "WEAVING", severity = "ORANGE"
            )
        )
        val encounters = listOf(
            AppearanceEncounter("MAKE", "Honda", 10, 3_000L),
            AppearanceEncounter("MAKE", "Acura", 4, 3_000L),
            AppearanceEncounter("BODY", "SUV", 8, 3_000L),
            AppearanceEncounter("BODY", "Sedan", 20, 3_000L),
            AppearanceEncounter("COLOR_BODY", "White SUV", 4, 3_000L)
        )

        val report = TrendsAnalyticsEngine.buildReport(logs, encounters)
        assertEquals(1, report.whiteSuvSpotlight.flaggedWhiteSuvs)
        assertEquals(4, report.whiteSuvSpotlight.observedWhiteSuvs)
        assertEquals(25.0f, report.whiteSuvSpotlight.whiteSuvRatePct, 0.01f)

        val hondaRow = report.watchlistRows.first { it.label == "Honda" }
        assertEquals(10.0f, hondaRow.ratePct, 0.01f)

        // Trusted-Group JSON serialization strips GPS coordinates for member privacy
        val json = TrustedGroupSync.serializeBundleJson("Unit-A1B2", logs)
        assertFalse(json.contains("41.7658"))
        val parsed = TrustedGroupSync.parseBundleJson(json, localUnitId = "Unit-XXXX")
        assertEquals("Unit-A1B2", parsed.sourceUnit)
        assertEquals(2, parsed.rows.size)
        assertEquals("Peer:Unit-A1B2", parsed.rows[0].reporterId)
        assertEquals(0.0, parsed.rows[0].latitude, 0.0001)
        assertEquals(0.90f, parsed.rows[0].matchConfidence, 0.001f)
    }
}
