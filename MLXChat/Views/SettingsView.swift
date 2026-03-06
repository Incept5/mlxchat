import SwiftUI

struct SettingsView: View {
    @State private var settings = SettingsManager.shared
    @State private var showingPromptPreview = false
    @FocusState private var focusedField: Field?

    private enum Field: Hashable {
        case systemPrompt
        case braveAPIKey
    }

    var body: some View {
        NavigationStack {
            List {
                if !settings.activeModelDownloads.isEmpty {
                    Section {
                        ForEach(settings.activeModelDownloads, id: \.hfId) { item in
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(settings.modelDisplayName(hfId: item.hfId))
                                        .font(.subheadline)
                                    Text("Downloading and preparing model...")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                Text("\(Int((item.progress * 100).rounded()))%")
                                    .font(.caption2)
                                    .monospacedDigit()
                                    .foregroundStyle(.secondary)
                                ProgressView(value: item.progress)
                                    .progressViewStyle(.circular)
                                    .frame(width: 18, height: 18)
                                Button {
                                    settings.cancelModelDownload(hfId: item.hfId)
                                } label: {
                                    Image(systemName: "xmark.circle.fill")
                                        .foregroundStyle(.secondary)
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    } header: {
                        Text("Active Downloads")
                    } footer: {
                        Text("Downloads continue in the background until they complete or you cancel them.")
                    }
                }

                Section("Currently Loaded") {
                    if let name = settings.loadedModelName {
                        HStack {
                            Image(systemName: "circle.fill")
                                .font(.caption2)
                                .foregroundStyle(.green)
                            Text(name)
                        }
                    } else {
                        Text("No model loaded")
                            .foregroundStyle(.secondary)
                    }
                }

                Section {
                    if settings.cachedModels.isEmpty {
                        Text("No models downloaded")
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(settings.cachedModels) { model in
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(model.displayName)
                                        .font(.subheadline)
                                    Text(model.id)
                                        .font(.caption2)
                                        .foregroundStyle(.tertiary)
                                }
                                Spacer()
                                if settings.loadedModelId == model.id {
                                    Text("Loaded")
                                        .font(.caption2)
                                        .foregroundStyle(.green)
                                        .padding(.trailing, 4)
                                }
                                Text(model.sizeFormatted)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        .onDelete { indexSet in
                            for index in indexSet {
                                settings.deleteModel(model: settings.cachedModels[index])
                            }
                        }

                        HStack {
                            Text("Total")
                                .font(.subheadline.bold())
                            Spacer()
                            Text(settings.totalCacheSize)
                                .font(.subheadline.bold())
                                .foregroundStyle(.secondary)
                        }
                    }
                } header: {
                    Text("Downloaded Models")
                } footer: {
                    Text("Swipe left to delete. Models will re-download when needed.")
                }

                Section {
                    Picker("GPU Memory Limit", selection: $settings.gpuMemoryLimitGB) {
                        ForEach(settings.gpuMemoryLimitOptions, id: \.self) { gb in
                            Text("\(gb) GB").tag(gb)
                        }
                    }

                    Picker("Model Download Limit", selection: $settings.maxModelDownloadSizeGB) {
                        ForEach(settings.modelDownloadLimitOptions, id: \.self) { gb in
                            if gb == 0 {
                                Text("No Limit").tag(gb)
                            } else {
                                Text("\(gb) GB").tag(gb)
                            }
                        }
                    }
                } header: {
                    Text("Memory")
                } footer: {
                    Text("GPU Memory Limit controls how much memory MLX may use at runtime and requires an app restart. Model Download Limit controls which models are offered for new downloads in the picker.")
                }

                Section {
                    Picker("Context Size", selection: $settings.contextSize) {
                        Text("4K").tag(4096)
                        Text("8K").tag(8192)
                        Text("12K").tag(12288)
                        Text("16K").tag(16384)
                        Text("20K").tag(20480)
                        Text("24K").tag(24576)
                        Text("28K").tag(28672)
                        Text("32K").tag(32768)
                    }

                    Toggle("Stream Responses", isOn: $settings.streamingEnabled)
                } header: {
                    Text("Generation")
                } footer: {
                    Text("Context window size affects maximum response length. Streaming shows tokens as they are generated.")
                }

                Section {
                    VStack(alignment: .leading) {
                        HStack {
                            Text("Max Image Dimension")
                            Spacer()
                            Text("\(settings.maxImageDimension) px")
                                .foregroundStyle(.secondary)
                        }
                        Slider(
                            value: Binding(
                                get: { Double(settings.maxImageDimension) },
                                set: { settings.maxImageDimension = Int($0) }
                            ),
                            in: 128...1024,
                            step: 64
                        )
                    }

                    VStack(alignment: .leading) {
                        HStack {
                            Text("JPEG Quality")
                            Spacer()
                            Text(String(format: "%.0f%%", settings.jpegQuality * 100))
                                .foregroundStyle(.secondary)
                        }
                        Slider(
                            value: $settings.jpegQuality,
                            in: 0.1...1.0,
                            step: 0.1
                        )
                    }
                } header: {
                    Text("Image Settings")
                } footer: {
                    Text("Smaller dimensions and lower quality reduce memory usage but may affect model accuracy.")
                }

                Section {
                    TextEditor(text: $settings.systemPrompt)
                        .font(.caption)
                        .frame(minHeight: 100)
                        .focused($focusedField, equals: .systemPrompt)

                    Button("Preview with Variables") {
                        focusedField = nil
                        showingPromptPreview = true
                    }

                    Button("Reset to Default") {
                        focusedField = nil
                        settings.systemPrompt = SettingsManager.defaultSystemPrompt
                    }
                    .foregroundStyle(.red)
                } header: {
                    Text("System Prompt")
                } footer: {
                    Text("Available variables: {today}, {date}, {time}, {datetime}, {timestamp}, {unixtime}, {location}, {address}, {coordinates}, {latitude}, {longitude}, {timezone}, {locale}, {device}, {system}, {version}, {username}")
                }

                Section {
                    Toggle("Enable Tools", isOn: $settings.toolsEnabled)

                    if settings.toolsEnabled {
                        SecureField("Brave Search API Key", text: $settings.braveAPIKey)
                            .textContentType(.password)
                            .autocorrectionDisabled()
                            .textInputAutocapitalization(.never)
                            .focused($focusedField, equals: .braveAPIKey)
                    }
                } header: {
                    Text("Tools")
                } footer: {
                    Text("Tools let the model search the web and fetch URLs. A Brave API key enables web search (get one free at brave.com/search/api).")
                }

                Section("Device Info") {
                    let totalRAM = Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)
                    let availableRAM = totalRAM * 0.6
                    MetricRow(label: "App Version", value: settings.appVersionString)
                    MetricRow(label: "Build Marker", value: settings.runtimeBuildMarker)
                    MetricRow(label: "Total RAM", value: String(format: "%.1f GB", totalRAM))
                    MetricRow(label: "Available for models", value: String(format: "%.1f GB", availableRAM))
                }
            }
            .navigationTitle("Settings")
            .scrollDismissesKeyboard(.interactively)
            .simultaneousGesture(
                TapGesture().onEnded {
                    focusedField = nil
                }
            )
            .onAppear {
                settings.refreshCachedModels()
            }
            .refreshable {
                focusedField = nil
                settings.refreshCachedModels()
            }
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") {
                        focusedField = nil
                    }
                }
            }
            .alert("System Prompt Preview", isPresented: $showingPromptPreview) {
                Button("OK") { }
            } message: {
                Text(settings.processedSystemPrompt())
            }
        }
    }
}

struct MetricRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
            Spacer()
            Text(value)
                .foregroundStyle(.secondary)
        }
    }
}
