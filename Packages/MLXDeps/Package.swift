// swift-tools-version: 6.1
import PackageDescription

// Local wrapper package that vends the MLX / HuggingFace stack to the app.
//
// WHY THIS EXISTS: mlx-swift-lm declares a default-on package trait
// `FoundationModelsIntegration` which builds the `MLXFoundationModels` adapter.
// That adapter tracks Apple's *beta* FoundationModels `SamplingMode` API, which
// does not compile against the installed iOS 27 beta SDK, and we don't use it.
// SwiftPM lets a dependent disable the trait with `traits: []` — but Xcode 27's
// project-level trait handling is unreliable (it re-enables the default trait on
// its implicit package re-resolution during `xcodebuild build`). Routing the
// dependency through this manifest makes the trait selection deterministic:
// SwiftPM honors `traits: []` here, so `MLXFoundationModels` is compiled to an
// empty library and never breaks the build.
//
// The app depends ONLY on the `MLXDeps` product and imports `MLXDeps`; the
// re-export shim (Sources/MLXDeps/Exports.swift) makes the underlying modules'
// symbols available. Keep the pins in sync with ToolTest/Package.swift.
let package = Package(
    name: "MLXDeps",
    platforms: [
        .iOS("27.0"),
        .macOS("14.0"),
    ],
    products: [
        .library(name: "MLXDeps", targets: ["MLXDeps"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main", traits: []),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.6"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.9"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
    ],
    targets: [
        .target(
            name: "MLXDeps",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ]
        ),
    ]
)
