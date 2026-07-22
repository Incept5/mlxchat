// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "ToolTest",
    platforms: [
        .macOS(.v14),
    ],
    dependencies: [
        // Disable the default-on `FoundationModelsIntegration` trait: it builds
        // the `MLXFoundationModels` adapter, whose FoundationModels `SamplingMode`
        // matching doesn't compile against the iOS/macOS 27 SDK — and we don't use
        // Apple's FoundationModels adapter. Turning the trait off compiles it to an
        // empty library and drops the dependency. Keep this in sync with the iOS
        // app's package trait config (project.yml).
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main", traits: []),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.9"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
    ],
    targets: [
        .executableTarget(
            name: "ToolTest",
            dependencies: [
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources",
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"]),
            ]
        ),
    ]
)
