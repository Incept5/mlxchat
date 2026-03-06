import Foundation

struct ModelSpec: Identifiable, Sendable {
    let id: String          // Short ID e.g. "qwen3.5-0.8b-q3"
    let hfId: String        // HuggingFace repo ID
    let displayName: String
    let family: String      // "0.8B", "2B", "4B"
    let quantization: String // "Q3", "Q4", "Q8"
    let sizeGB: Double
    let isThinkingModel: Bool = true
    let supportsNoThink: Bool = true
    let supportsVision: Bool = true
}

struct ModelRegistry {
    static let models: [ModelSpec] = [
        // 0.8B family
        ModelSpec(
            id: "qwen3.5-0.8b-q4",
            hfId: "mlx-community/Qwen3.5-0.8B-MLX-4bit",
            displayName: "Qwen3.5 0.8B Q4",
            family: "0.8B",
            quantization: "Q4",
            sizeGB: 0.7
        ),
        ModelSpec(
            id: "qwen3.5-0.8b-q8",
            hfId: "mlx-community/Qwen3.5-0.8B-MLX-8bit",
            displayName: "Qwen3.5 0.8B Q8",
            family: "0.8B",
            quantization: "Q8",
            sizeGB: 1.1
        ),
        ModelSpec(
            id: "qwen3.5-0.8b-bf16",
            hfId: "mlx-community/Qwen3.5-0.8B-MLX-bf16",
            displayName: "Qwen3.5 0.8B BF16",
            family: "0.8B",
            quantization: "BF16",
            sizeGB: 1.6
        ),
        // 2B family
        ModelSpec(
            id: "qwen3.5-2b-q4",
            hfId: "mlx-community/Qwen3.5-2B-MLX-4bit",
            displayName: "Qwen3.5 2B Q4",
            family: "2B",
            quantization: "Q4",
            sizeGB: 1.8
        ),
        ModelSpec(
            id: "qwen3.5-2b-q8",
            hfId: "mlx-community/Qwen3.5-2B-MLX-8bit",
            displayName: "Qwen3.5 2B Q8",
            family: "2B",
            quantization: "Q8",
            sizeGB: 2.8
        ),
        ModelSpec(
            id: "qwen3.5-2b-bf16",
            hfId: "mlx-community/Qwen3.5-2B-MLX-bf16",
            displayName: "Qwen3.5 2B BF16",
            family: "2B",
            quantization: "BF16",
            sizeGB: 4.1
        ),
        // 4B family
        ModelSpec(
            id: "qwen3.5-4b-q4",
            hfId: "mlx-community/Qwen3.5-4B-MLX-4bit",
            displayName: "Qwen3.5 4B Q4",
            family: "4B",
            quantization: "Q4",
            sizeGB: 2.8
        ),
        ModelSpec(
            id: "qwen3.5-4b-q8",
            hfId: "mlx-community/Qwen3.5-4B-MLX-8bit",
            displayName: "Qwen3.5 4B Q8",
            family: "4B",
            quantization: "Q8",
            sizeGB: 5.6
        ),
        // 9B family
        ModelSpec(
            id: "qwen3.5-9b-q4",
            hfId: "mlx-community/Qwen3.5-9B-MLX-4bit",
            displayName: "Qwen3.5 9B Q4",
            family: "9B",
            quantization: "Q4",
            sizeGB: 5.6
        ),
    ]

    /// Group models by family for picker display.
    static var groupedByFamily: [(family: String, models: [ModelSpec])] {
        let families = ["0.8B", "2B", "4B", "9B"]
        return families.compactMap { family in
            let group = models.filter { $0.family == family }
            return group.isEmpty ? nil : (family: family, models: group)
        }
    }

    static func find(id: String) -> ModelSpec? {
        models.first { $0.id == id }
    }

    static func find(hfId: String) -> ModelSpec? {
        models.first { $0.hfId == hfId }
    }

    /// Filter models that fit in available memory.
    static func modelsForMemory(availableGB: Double) -> [ModelSpec] {
        models.filter { $0.sizeGB <= availableGB }
    }
}
