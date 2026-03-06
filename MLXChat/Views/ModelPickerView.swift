import SwiftUI

struct ModelPickerView: View {
    @Bindable var viewModel: ChatViewModel
    @State private var settings = SettingsManager.shared
    @Environment(\.dismiss) private var dismiss

    private var groupedModels: [(family: String, models: [ModelSpec])] {
        ModelRegistry.groupedByFamily.map { group in
            (group.family, group.models)
        }
    }

    var body: some View {
        NavigationStack {
            List {
                ForEach(groupedModels, id: \.family) { group in
                    Section(group.family) {
                        ForEach(group.models) { model in
                            if let progress = settings.modelDownloadProgress[model.hfId] {
                                HStack {
                                    modelRowContent(model)
                                    downloadAccessory(hfId: model.hfId, progress: progress)
                                }
                            } else {
                                Button {
                                    selectModel(model)
                                } label: {
                                    HStack {
                                        modelRowContent(model)
                                        staticAccessory(for: model)
                                    }
                                }
                                .disabled(isHiddenByDownloadLimit(model))
                            }
                        }
                    }
                }
            }
            .navigationTitle("Select Model")
            .navigationBarTitleDisplayMode(.inline)
            .safeAreaInset(edge: .bottom) {
                if settings.maxModelDownloadSizeGB > 0 {
                    Text("New downloads are limited to \(settings.maxModelDownloadSizeGB) GB or smaller. Already-downloaded models remain selectable.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal)
                        .padding(.vertical, 8)
                        .background(.thinMaterial)
                }
            }
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
            .onAppear {
                settings.refreshCachedModels()
            }
        }
    }

    private func selectModel(_ model: ModelSpec) {
        viewModel.selectedModel = model

        if settings.isModelDownloaded(hfId: model.hfId) {
            // Unload old engine so new model gets loaded on next send
            if let engine = settings.sharedEngine {
                Task {
                    await engine.unloadModel()
                    settings.sharedEngine = nil
                }
            }

            dismiss()
            return
        }

        settings.setModelDownloadProgress(hfId: model.hfId, progress: 0.01)

        let task = Task {
            do {
                let engine = try await downloadModel(model)
                try Task.checkCancellation()
                settings.sharedEngine = engine
                settings.refreshCachedModels()
                settings.setModelDownloadProgress(hfId: model.hfId, progress: 1)
                settings.finishModelDownloadTask(hfId: model.hfId)
                dismiss()
            } catch {
                settings.setModelDownloadProgress(hfId: model.hfId, progress: 0)
                settings.finishModelDownloadTask(hfId: model.hfId)
            }
        }
        settings.registerModelDownloadTask(hfId: model.hfId, task: task)
    }

    private func downloadModel(_ model: ModelSpec) async throws -> MLXEngine {
        let engine = settings.sharedEngine ?? MLXEngine(memoryLimitGB: settings.gpuMemoryLimitGB)
        let maxAttempts = 2

        for attempt in 1...maxAttempts {
            do {
                _ = try await engine.loadModel(
                    id: model.hfId,
                    forVision: false,
                    progress: { progress in
                        let fraction = progress.fractionCompleted
                        Task { @MainActor in
                            settings.setModelDownloadProgress(
                                hfId: model.hfId,
                                progress: fraction.isFinite ? fraction : 0.01
                            )
                        }
                    }
                )
                return engine
            } catch {
                try Task.checkCancellation()
                guard attempt < maxAttempts else { throw error }
                settings.setModelDownloadProgress(
                    hfId: model.hfId,
                    progress: max(settings.modelDownloadProgress[model.hfId] ?? 0.01, 0.01)
                )
                try await Task.sleep(for: .seconds(1))
            }
        }

        return engine
    }

    private func isHiddenByDownloadLimit(_ model: ModelSpec) -> Bool {
        settings.maxModelDownloadSizeGB > 0 &&
            !settings.isModelDownloaded(hfId: model.hfId) &&
            model.sizeGB > Double(settings.maxModelDownloadSizeGB)
    }

    @ViewBuilder
    private func modelRowContent(_ model: ModelSpec) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(model.displayName)
                .font(.subheadline)
                .foregroundStyle(.primary)
            if settings.modelDownloadProgress[model.hfId] != nil {
                Text("Downloading and preparing model...")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else {
                Text(model.hfId)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }

        Spacer()

        Text(model.quantization)
            .font(.caption2)
            .fontWeight(.semibold)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(quantColor(model.quantization).opacity(0.15))
            .foregroundStyle(quantColor(model.quantization))
            .clipShape(Capsule())
            .fixedSize()

        Text(String(format: "%.1f GB", model.sizeGB))
            .font(.caption)
            .foregroundStyle(.secondary)
            .frame(width: 58, alignment: .trailing)
    }

    @ViewBuilder
    private func staticAccessory(for model: ModelSpec) -> some View {
        if settings.isModelDownloaded(hfId: model.hfId) {
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .font(.caption)
        }

        if isHiddenByDownloadLimit(model) {
            Text("Limit")
                .font(.caption2)
                .foregroundStyle(.orange)
        }

        if settings.loadedModelId == model.hfId {
            Image(systemName: "circle.fill")
                .foregroundStyle(.blue)
                .font(.caption2)
        }
    }

    private func downloadAccessory(hfId: String, progress: Double) -> some View {
        HStack(spacing: 8) {
            Text("\(Int((progress * 100).rounded()))%")
                .font(.caption2)
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .frame(width: 32, alignment: .trailing)

            ProgressView(value: progress)
                .progressViewStyle(.circular)
                .frame(width: 18, height: 18)

            Button {
                settings.cancelModelDownload(hfId: hfId)
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
        }
    }

    private func quantColor(_ quant: String) -> Color {
        switch quant {
        case "Q3": return .orange
        case "Q4": return .blue
        case "Q8": return .green
        case "BF16": return .red
        case "VLM 8": return .purple
        default: return .secondary
        }
    }
}
