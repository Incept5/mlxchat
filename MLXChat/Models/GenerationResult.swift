import Foundation

struct GenerationMetrics: Sendable {
    var promptTokenCount: Int = 0
    var generationTokenCount: Int = 0
    var promptTimeSeconds: Double = 0
    var generateTimeSeconds: Double = 0
    var totalTimeSeconds: Double = 0
    var tokensPerSecond: Double = 0
    var promptTokensPerSecond: Double = 0
    var peakMemoryMB: Double = 0
    var baselineMemoryMB: Double = 0
}

/// Outcome of the on-device model benchmark ("Test Model"). Captures throughput,
/// memory and a thermal reading taken around a single fixed generation run.
struct ModelTestResult: Sendable {
    var modelName: String = ""
    var metrics = GenerationMetrics()
    /// `ProcessInfo.ThermalState` sampled just after the run — the closest live
    /// proxy iOS exposes for power/energy draw (there is no public watts API).
    var thermalState: ProcessInfo.ThermalState = .nominal
    var error: String?

    var thermalLabel: String {
        switch thermalState {
        case .nominal: "Nominal"
        case .fair: "Fair"
        case .serious: "Serious"
        case .critical: "Critical"
        @unknown default: "Unknown"
        }
    }
}

struct GenerationResult: Sendable {
    var output: String = ""
    var cleanedOutput: String = ""
    var thinkingText: String = ""
    var metrics = GenerationMetrics()
    var error: String?

    var success: Bool {
        error == nil && !output.isEmpty
    }
}
