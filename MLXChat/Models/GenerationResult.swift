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

struct GenerationResult: Sendable {
    var output: String = ""
    var cleanedOutput: String = ""
    var metrics = GenerationMetrics()
    var error: String?

    var success: Bool {
        error == nil && !output.isEmpty
    }
}
