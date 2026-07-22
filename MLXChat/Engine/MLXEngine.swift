import Foundation
import MLXDeps

actor MLXEngine {

    private var modelContainer: ModelContainer?
    private var currentModelId: String?
    private var isVLM = false

    /// A `Tokenizers` tokenizer kept around only for the Qwen3.5 non-thinking
    /// template patch (see ``qwen35PatchIfNeeded``). The 3.x model context exposes
    /// a minimal `MLXLMCommon.Tokenizer` that can't apply a custom template
    /// literal, so we load a `PreTrainedTokenizer` directly. Cached per model id
    /// to avoid re-parsing tokenizer.json on every turn.
    private var cachedPatchTokenizer: (id: String, tokenizer: any Tokenizers.Tokenizer)?

    /// Everything needed to tokenise with the patched Qwen3.5 template.
    private struct Qwen35Patch: Sendable {
        let tokenizer: any Tokenizers.Tokenizer
        let template: String
    }

    /// Root of the on-disk model cache (Python-compatible HubCache layout:
    /// `<root>/models--<org>--<repo>/snapshots/<commit>/…`).
    ///
    /// mlx-swift-lm 3.x downloads through a `HubClient`, whose default cache in a
    /// sandboxed app is `Library/Caches/huggingface/hub` — and iOS purges
    /// `Library/Caches` under storage pressure, so a previously-loaded model
    /// silently disappears and has to be re-downloaded. We pin the cache to
    /// `Documents/huggingface` instead (non-purgeable, and the location
    /// `SettingsManager` scans for downloaded models).
    static let cacheRoot: URL = {
        let documents = FileManager.default
            .urls(for: .documentDirectory, in: .userDomainMask).first!
        return documents.appending(component: "huggingface")
    }()

    /// Shared Hub client backing every download, pinned to ``cacheRoot``.
    static let hubClient = HubClient(cache: HubCache(cacheDirectory: cacheRoot))

    init(memoryLimitGB: Int = 5) {
        GPU.set(memoryLimit: memoryLimitGB * 1024 * 1024 * 1024)
    }

    func loadModel(
        id: String,
        forVision: Bool = false,
        progress: (@Sendable (Progress) -> Void)? = nil
    ) async throws -> Double {
        // If already loaded with the right factory, skip
        if currentModelId == id && (forVision == isVLM || !forVision) { return 0 }

        // Unload any existing model first
        if modelContainer != nil {
            unloadModel()
        }

        let start = CFAbsoluteTimeGetCurrent()

        // mlx-swift-lm 3.x: tokenizers load through a TokenizerLoader; the macro
        // wraps AutoTokenizer in the MLXLMCommon protocol.
        let tokenizerLoader = #huggingFaceTokenizerLoader()

        // Fast path: if a complete copy of the model is already on disk (sideloaded
        // over USB, or downloaded on a previous run) load straight from its snapshot
        // directory. This bypasses the HubClient entirely — no network resolution,
        // no re-download, no offline hang. Essential for sideloaded models: their
        // weights aren't stored as Hub blobs, so the id-based path treats
        // `model.safetensors` as "missing" and re-downloads the whole thing.
        if let localDir = Self.completeLocalSnapshot(forModelId: id) {
            if forVision {
                modelContainer = try await VLMModelFactory.shared.loadContainer(
                    from: localDir, using: tokenizerLoader)
                isVLM = true
            } else {
                do {
                    modelContainer = try await LLMModelFactory.shared.loadContainer(
                        from: localDir, using: tokenizerLoader)
                    isVLM = false
                } catch {
                    modelContainer = try await VLMModelFactory.shared.loadContainer(
                        from: localDir, using: tokenizerLoader)
                    isVLM = true
                }
            }
            currentModelId = id
            return CFAbsoluteTimeGetCurrent() - start
        }

        // Not on disk — download via the Hub. Downloads go through a Downloader
        // (our Documents-pinned HubClient).
        let config = ModelConfiguration(
            id: id,
            // Qwen3.5 tool use maps to the qwen3_coder-style XML function format.
            toolCallFormat: Self.toolCallFormat(for: id)
        )
        let progressHandler: @Sendable (Progress) -> Void = progress ?? { _ in }
        let downloader = #hubDownloader(Self.hubClient)

        if forVision {
            // Vision requests require VLMModelFactory
            modelContainer = try await VLMModelFactory.shared.loadContainer(
                from: downloader, using: tokenizerLoader,
                configuration: config, progressHandler: progressHandler)
            isVLM = true
        } else {
            // Text-only: try LLM first (2x faster, better tool support), fall back to VLM
            do {
                modelContainer = try await LLMModelFactory.shared.loadContainer(
                    from: downloader, using: tokenizerLoader,
                    configuration: config, progressHandler: progressHandler)
                isVLM = false
            } catch {
                modelContainer = try await VLMModelFactory.shared.loadContainer(
                    from: downloader, using: tokenizerLoader,
                    configuration: config, progressHandler: progressHandler)
                isVLM = true
            }
        }

        currentModelId = id
        return CFAbsoluteTimeGetCurrent() - start
    }

    /// Returns the on-disk snapshot directory for `id` **iff** a complete copy is
    /// present: a `snapshots/<hash>/` folder holding `config.json` and its weights
    /// (single `model.safetensors`, or every shard named by
    /// `model.safetensors.index.json`). Filesystem-direct — it doesn't rely on Hub
    /// `refs`/blobs — so it recognises a freshly sideloaded model the HubClient has
    /// never reconciled. An incomplete copy returns `nil`, falling back to download.
    static func completeLocalSnapshot(forModelId id: String) -> URL? {
        let fm = FileManager.default
        let repoDir = cacheRoot.appending(
            component: "models--" + id.replacingOccurrences(of: "/", with: "--"))
        let snapshotsDir = repoDir.appending(component: "snapshots")
        guard let snapshots = try? fm.contentsOfDirectory(
            at: snapshotsDir, includingPropertiesForKeys: [.isDirectoryKey]) else { return nil }

        for snapshot in snapshots {
            let files = Set((try? fm.contentsOfDirectory(atPath: snapshot.path)) ?? [])
            guard files.contains("config.json") else { continue }

            // Sharded model: require every shard the index references.
            if files.contains("model.safetensors.index.json"),
               let index = try? Data(contentsOf: snapshot.appending(component: "model.safetensors.index.json")),
               let json = try? JSONSerialization.jsonObject(with: index) as? [String: Any],
               let weightMap = json["weight_map"] as? [String: String] {
                let shards = Set(weightMap.values)
                if !shards.isEmpty, shards.isSubset(of: files) { return snapshot }
            }

            // Single-file weights.
            if files.contains("model.safetensors") { return snapshot }
        }
        return nil
    }

    var loadedAsVLM: Bool { isVLM }

    /// Whether a model is currently resident and ready to generate.
    var isModelLoaded: Bool { modelContainer != nil }

    /// The HF id of the currently-loaded model, if any.
    var loadedModelId: String? { currentModelId }

    func clearCache() {
        Memory.cacheLimit = 0
        Memory.clearCache()
        Memory.cacheLimit = 512 * 1024 * 1024
    }

    func generateChat(
        messages: [Chat.Message],
        maxTokens: Int = 500,
        temperature: Float = 0.6,
        topP: Float = 0.95,
        topK: Int = 20,
        minP: Float = 0.0,
        repetitionPenalty: Float? = nil,
        presencePenalty: Float? = nil,
        enableThinking: Bool? = nil,
        hasImage: Bool = false,
        tools: [ToolSpec]? = nil,
        toolDispatch: (@Sendable (String, [String: String]) async -> (name: String, result: String))? = nil,
        onToolCall: (@MainActor @Sendable (String) -> Void)? = nil,
        onToolResult: (@MainActor @Sendable (String, String) -> Void)? = nil,
        onChunk: (@MainActor @Sendable (String) -> Void)? = nil
    ) async throws -> GenerationResult {
        guard let container = modelContainer else {
            var result = GenerationResult()
            result.error = "No model loaded"
            return result
        }

        // Drop any pooled buffers *before* generation starts (no command buffer is
        // in flight yet, so this is safe) to keep peak memory down, then re-enable a
        // recycle pool for the generation itself. The pool MUST stay non-zero while
        // generating: with cacheLimit == 0, an MLXArray released while the GPU still
        // has an in-flight command buffer referencing it frees the underlying
        // MTLBuffer out from under that command buffer — Metal then aborts with
        // "command buffer references deallocated object". This bit only the image
        // path, which previously left the pool disabled throughout (the early-cancel
        // in the stream loop tears generation down mid async-eval). Images get a
        // smaller pool than text to keep VLM peak memory in check.
        Memory.cacheLimit = 0
        Memory.clearCache()
        Memory.cacheLimit = hasImage ? 256 * 1024 * 1024 : 512 * 1024 * 1024
        let baselineMemory = Double(Memory.activeMemory) / (1024 * 1024)
        Memory.peakMemory = 0

        var additionalCtx: [String: any Sendable]?
        if let enableThinking {
            additionalCtx = ["enable_thinking": enableThinking]
        }

        var currentMessages = messages

        // Original HF id (e.g. "mlx-community/Qwen3.5-4B-MLX-4bit"). After a 3.x
        // load, `context.configuration` is rebuilt as a local `.directory`, so its
        // `name` is the snapshot path rather than the repo id — we carry the real
        // id here so Qwen3.5 detection for the thinking-template patch still works.
        let modelId = currentModelId

        // For Qwen3.5 with thinking off, build the patched-template tokeniser up
        // front (cached) so `prepareInput` can use it inside the perform closure.
        let qwen35Patch = await qwen35PatchIfNeeded(
            modelId: modelId, enableThinking: enableThinking, hasImage: hasImage)

        var generateParams = GenerateParameters(
            temperature: temperature,
            topP: topP,
            topK: topK,
            minP: minP
        )
        if let repetitionPenalty {
            generateParams.repetitionPenalty = repetitionPenalty
            generateParams.repetitionContextSize = 64
        }
        // Qwen3.5 curbs repetition with a non-zero presence_penalty (~1.5) while
        // leaving repetition_penalty at 1.0 (off). Use a wide window to approximate
        // the OpenAI-style whole-sequence presence penalty the model card assumes;
        // mlx's default window of 20 tokens is too short to match that intent.
        if let presencePenalty {
            generateParams.presencePenalty = presencePenalty
            generateParams.presenceContextSize = 256
        }

        let promptStart = CFAbsoluteTimeGetCurrent()
        var finalOutput = ""
        var finalInfo: GenerateCompletionInfo?
        var toolCallCount = 0
        var lastToolResult: String?
        let maxToolCalls = 5

        // Tool-call loop: generate, parse tool calls from text, execute, re-generate
        while toolCallCount < maxToolCalls {
            let loopResult: (text: String, info: GenerateCompletionInfo?, toolCall: ToolCall?) =
                try await container.perform(values: (
                    messages: currentMessages,
                    tools: tools,
                    additionalCtx: additionalCtx,
                    generateParams: generateParams,
                    maxTokens: maxTokens
                )) { context, values in
                    let input = try await Self.prepareInput(
                        context: context,
                        qwen35Patch: qwen35Patch,
                        messages: values.messages,
                        tools: values.tools,
                        additionalContext: values.additionalCtx,
                        hasImage: hasImage
                    )
                    let iterator = try TokenIterator(
                        input: input,
                        model: context.model,
                        parameters: values.generateParams
                    )
                    let (stream, task) = MLXLMCommon.generateTask(
                        promptTokenCount: input.text.tokens.size,
                        modelConfiguration: context.configuration,
                        tokenizer: context.tokenizer,
                        iterator: iterator
                    )

                    var generatedText = ""
                    var completionInfo: GenerateCompletionInfo?
                    var detectedToolCall: ToolCall?
                    var cancelledEarly = false
                    var chunkCount = 0
                    let thinkingDeliberate =
                        (values.additionalCtx?["enable_thinking"] as? Bool) == true

                    for await generation in stream {
                        try Task.checkCancellation()

                        switch generation {
                        case .chunk(let chunk):
                            guard detectedToolCall == nil else { continue }
                            generatedText += chunk
                            chunkCount += 1
                            if let onChunk, !chunk.isEmpty {
                                await onChunk(chunk)
                            }
                            let stuckInThink =
                                !thinkingDeliberate &&
                                ThinkingText.isOpen(generatedText) &&
                                generatedText.count >= min(values.maxTokens, 160)
                            if chunkCount >= values.maxTokens
                                || stuckInThink
                                || (generatedText.count >= 60 && Self.hasRepetition(generatedText))
                            {
                                cancelledEarly = true
                                task.cancel()
                                break
                            }
                        case .info(let info):
                            completionInfo = info
                        case .toolCall(let toolCall):
                            detectedToolCall = toolCall
                            task.cancel()
                        }
                    }

                    if cancelledEarly || detectedToolCall != nil {
                        task.cancel()
                    }
                    await task.value
                    return (generatedText, completionInfo, detectedToolCall)
                }

            var generatedText = loopResult.text
            var completionInfo = loopResult.info
            var detectedToolCall = loopResult.toolCall

            if detectedToolCall == nil, tools != nil {
                detectedToolCall = Self.parseToolCall(from: generatedText).map(Self.makeToolCall(from:))
            }

            if let call = detectedToolCall, let dispatch = toolDispatch {
                let assistantText = generatedText.trimmingCharacters(in: .whitespacesAndNewlines)
                if !assistantText.isEmpty {
                    currentMessages.append(.assistant(assistantText))
                }

                // Notify UI
                if let onToolCall {
                    await onToolCall(call.function.name)
                }

                // Execute the tool
                let toolResult = await dispatch(call.function.name, Self.stringArguments(from: call))

                // Append tool result
                currentMessages.append(.tool(toolResult.result))
                lastToolResult = toolResult.result
                if let onToolResult {
                    await onToolResult(toolResult.name, toolResult.result)
                }

                toolCallCount += 1
                finalInfo = completionInfo
                continue
            }

            // No tool call — this is the final response
            var output = generatedText
            // Trim repetition from final output
            output = Self.trimRepetition(output)
            finalOutput = output
            // Always use the final round's info for accurate tok/s
            finalInfo = completionInfo
            break
        }

        let totalTime = CFAbsoluteTimeGetCurrent() - promptStart
        let peakMemory = Double(Memory.peakMemory) / (1024 * 1024)

        var result = GenerationResult()
        result.output = finalOutput
        let (thinking, cleaned) = ThinkingText.split(finalOutput)
        result.thinkingText = thinking
        // If the model's response is empty after stripping but a tool returned data, use that
        if cleaned.isEmpty, let toolResult = lastToolResult {
            result.cleanedOutput = toolResult
        } else {
            result.cleanedOutput = cleaned
        }

        if let info = finalInfo {
            result.metrics.promptTokenCount = info.promptTokenCount
            result.metrics.generationTokenCount = info.generationTokenCount
            result.metrics.promptTimeSeconds = info.promptTime
            result.metrics.generateTimeSeconds = info.generateTime
            result.metrics.tokensPerSecond = info.tokensPerSecond
            result.metrics.promptTokensPerSecond = info.promptTokensPerSecond
        } else {
            result.metrics.generateTimeSeconds = totalTime
        }

        result.metrics.totalTimeSeconds = totalTime
        result.metrics.peakMemoryMB = peakMemory
        result.metrics.baselineMemoryMB = baselineMemory

        return result
    }

    func unloadModel() {
        modelContainer = nil
        currentModelId = nil
        cachedPatchTokenizer = nil
        Memory.cacheLimit = 0
        Memory.clearCache()
    }

    private static func toolCallFormat(for modelId: String) -> ToolCallFormat? {
        let normalized = modelId.lowercased()
        if normalized.contains("qwen3.5") || normalized.contains("qwen3_5") {
            return .xmlFunction
        }
        return nil
    }

    private static func prepareInput(
        context: ModelContext,
        qwen35Patch: Qwen35Patch?,
        messages: [Chat.Message],
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?,
        hasImage: Bool
    ) async throws -> LMInput {
        if let qwen35Patch {
            let rawMessages = DefaultMessageGenerator().generate(messages: messages)
            let promptTokens = try qwen35Patch.tokenizer.applyChatTemplate(
                messages: rawMessages,
                chatTemplate: .literal(qwen35Patch.template),
                addGenerationPrompt: true,
                truncation: false,
                maxLength: nil,
                tools: tools,
                additionalContext: additionalContext
            )
            return LMInput(tokens: MLXArray(promptTokens))
        }

        let userInput = UserInput(
            chat: messages,
            tools: tools,
            additionalContext: additionalContext
        )
        return try await context.processor.prepare(input: userInput)
    }

    /// Builds the patched-template tokeniser for a Qwen3.5 turn with thinking
    /// off, or returns `nil` when the patch doesn't apply. The tokeniser is
    /// cached per model id (``cachedPatchTokenizer``) so tokenizer.json is only
    /// parsed once per loaded model.
    private func qwen35PatchIfNeeded(
        modelId: String?,
        enableThinking: Bool?,
        hasImage: Bool
    ) async -> Qwen35Patch? {
        guard
            !hasImage,
            let modelId, Self.isQwen35(modelId),
            enableThinking == false,
            let modelDirectory = Self.snapshotDirectory(forModelId: modelId),
            let rawTemplate = try? String(
                contentsOf: modelDirectory.appending(path: "chat_template.jinja"),
                encoding: .utf8)
        else {
            return nil
        }

        let template = Self.patchQwen35NonThinkingTemplate(rawTemplate)

        if let cached = cachedPatchTokenizer, cached.id == modelId {
            return Qwen35Patch(tokenizer: cached.tokenizer, template: template)
        }
        guard let tokenizer = try? await AutoTokenizer.from(modelFolder: modelDirectory) else {
            return nil
        }
        cachedPatchTokenizer = (modelId, tokenizer)
        return Qwen35Patch(tokenizer: tokenizer, template: template)
    }

    /// Resolves the local snapshot directory for a downloaded model id via the
    /// shared HubCache (the parent of its `chat_template.jinja`).
    private static func snapshotDirectory(forModelId id: String) -> URL? {
        guard
            let cache = hubClient.cache,
            let repo = Repo.ID(rawValue: id),
            let templateFile = cache.cachedFilePath(
                repo: repo, kind: .model, revision: "main",
                filename: "chat_template.jinja")
        else {
            return nil
        }
        return templateFile.deletingLastPathComponent()
    }

    private static func patchQwen35NonThinkingTemplate(_ chatTemplate: String) -> String {
        let brokenBlock = """
        {%- if enable_thinking is defined and enable_thinking is false %}
                {{- '<think>\\n\\n</think>\\n\\n' }}
            {%- else %}
                {{- '<think>\\n' }}
            {%- endif %}
        """
        let fixedBlock = """
        {%- if enable_thinking is defined and enable_thinking is false %}
                {{- '' }}
            {%- else %}
                {{- '<think>\\n' }}
            {%- endif %}
        """

        if chatTemplate.contains(brokenBlock) {
            return chatTemplate.replacingOccurrences(of: brokenBlock, with: fixedBlock)
        }

        return chatTemplate.replacingOccurrences(
            of: "{{- '<think>\\n\\n</think>\\n\\n' }}",
            with: "{{- '' }}"
        )
    }

    private static func isQwen35(_ name: String) -> Bool {
        let normalized = name.lowercased()
        return normalized.contains("qwen3.5") || normalized.contains("qwen3_5")
    }

    // MARK: - Tool Call Parsing (from decoded text, matching benchmark approach)

    private struct ParsedToolCall {
        let name: String
        let arguments: [String: String]
    }

    /// Parse tool calls from generated text using multiple format patterns
    private static func parseToolCall(from text: String) -> ParsedToolCall? {
        // Pattern 1: Full format: <tool_call><function=name>...params...</function></tool_call>
        let qwenPattern = #"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>"#
        if let regex = try? NSRegularExpression(pattern: qwenPattern, options: [.dotMatchesLineSeparators]) {
            let range = NSRange(text.startIndex..., in: text)
            if let match = regex.firstMatch(in: text, range: range),
               let nameRange = Range(match.range(at: 1), in: text),
               let bodyRange = Range(match.range(at: 2), in: text) {
                let name = String(text[nameRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                let body = String(text[bodyRange])
                let args = parseXMLParameters(body)
                return ParsedToolCall(name: name, arguments: args)
            }
        }

        // Pattern 2: <function=name>...params...</function> without wrapper
        let xmlFuncPattern = #"<function=([^>]+)>(.*?)</function>"#
        if let regex = try? NSRegularExpression(pattern: xmlFuncPattern, options: [.dotMatchesLineSeparators]) {
            let range = NSRange(text.startIndex..., in: text)
            if let match = regex.firstMatch(in: text, range: range),
               let nameRange = Range(match.range(at: 1), in: text),
               let bodyRange = Range(match.range(at: 2), in: text) {
                let name = String(text[nameRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                let body = String(text[bodyRange])
                let args = parseXMLParameters(body)
                return ParsedToolCall(name: name, arguments: args)
            }
        }

        // Pattern 3: <tool_call> with just function name, no </function> closing
        // e.g. <tool_call>\n<function=date_time>\n</tool_call>
        let loosePattern = #"<tool_call>\s*<function=([a-z_]+)\s*>?\s*[\s\S]*?</tool_call>"#
        if let regex = try? NSRegularExpression(pattern: loosePattern, options: [.dotMatchesLineSeparators]) {
            let range = NSRange(text.startIndex..., in: text)
            if let match = regex.firstMatch(in: text, range: range),
               let nameRange = Range(match.range(at: 1), in: text) {
                let name = String(text[nameRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                // Extract any parameters between the tags
                let fullRange = Range(match.range, in: text)!
                let body = String(text[fullRange])
                let args = parseXMLParameters(body)
                return ParsedToolCall(name: name, arguments: args)
            }
        }

        // Pattern 4: Just <function=name> anywhere (no closing tag, model stopped)
        let barePattern = #"<function=([a-z_]+)\s*/?>"#
        if let regex = try? NSRegularExpression(pattern: barePattern, options: []) {
            let range = NSRange(text.startIndex..., in: text)
            if let match = regex.firstMatch(in: text, range: range),
               let nameRange = Range(match.range(at: 1), in: text) {
                let name = String(text[nameRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                let args = parseXMLParameters(text)
                return ParsedToolCall(name: name, arguments: args)
            }
        }

        // Pattern 5: JSON in <tool_call> tags
        let jsonToolCallPattern = #"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>"#
        if let regex = try? NSRegularExpression(pattern: jsonToolCallPattern, options: []) {
            let range = NSRange(text.startIndex..., in: text)
            if let match = regex.firstMatch(in: text, range: range),
               let jsonRange = Range(match.range(at: 1), in: text) {
                let jsonStr = String(text[jsonRange])
                if let data = jsonStr.data(using: .utf8),
                   let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let name = dict["name"] as? String {
                    let argsDict = dict["arguments"] as? [String: Any] ?? [:]
                    var args: [String: String] = [:]
                    for (k, v) in argsDict {
                        args[k] = "\(v)"
                    }
                    return ParsedToolCall(name: name, arguments: args)
                }
            }
        }

        return nil
    }

    private static func parseXMLParameters(_ body: String) -> [String: String] {
        var args: [String: String] = [:]
        let paramPattern = #"<parameter=([^>]+)>([\s\S]*?)</parameter>"#
        if let paramRegex = try? NSRegularExpression(pattern: paramPattern, options: [.dotMatchesLineSeparators]) {
            let paramRange = NSRange(body.startIndex..., in: body)
            let paramMatches = paramRegex.matches(in: body, range: paramRange)
            for pm in paramMatches {
                if let keyRange = Range(pm.range(at: 1), in: body),
                   let valRange = Range(pm.range(at: 2), in: body) {
                    let key = String(body[keyRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                    let value = String(body[valRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                    args[key] = value
                }
            }
        }
        return args
    }

    private static func makeToolCall(from parsed: ParsedToolCall) -> ToolCall {
        ToolCall(function: .init(name: parsed.name, arguments: parsed.arguments))
    }

    private static func stringArguments(from toolCall: ToolCall) -> [String: String] {
        var args: [String: String] = [:]
        for (key, value) in toolCall.function.arguments {
            args[key] = value.stringValue ?? "\(value.anyValue)"
        }
        return args
    }

    // MARK: - Repetition Detection

    /// Check if text has repeating patterns
    private static func hasRepetition(_ text: String) -> Bool {
        if hasRepeatedLine(text, minimumRepeats: 3) {
            return true
        }

        let checkLen = min(text.count, 600)
        guard checkLen >= 48 else { return false }
        let tail = String(text.suffix(checkLen))

        for patternLen in stride(from: 12, through: min(160, checkLen / 3), by: 4) {
            let pattern = String(tail.suffix(patternLen))
            var count = 0
            var remaining = tail
            while remaining.hasSuffix(pattern) {
                count += 1
                remaining = String(remaining.dropLast(patternLen))
            }
            if count >= 3 { return true }
        }
        return false
    }

    /// Trim repeating patterns from the end of text
    private static func trimRepetition(_ text: String) -> String {
        if let deduped = trimRepeatedLines(text, minimumRepeats: 3) {
            return deduped
        }

        let checkLen = min(text.count, 800)
        guard checkLen >= 48 else { return text }
        let tail = String(text.suffix(checkLen))

        for patternLen in stride(from: 12, through: min(160, checkLen / 3), by: 4) {
            let pattern = String(tail.suffix(patternLen))
            var count = 0
            var remaining = tail
            while remaining.hasSuffix(pattern) {
                count += 1
                remaining = String(remaining.dropLast(patternLen))
            }
            if count >= 3 {
                let fullLen = text.count
                let trimPoint = fullLen - (count * patternLen) + patternLen
                return String(text.prefix(trimPoint)).trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        return text
    }

    private static func hasRepeatedLine(_ text: String, minimumRepeats: Int) -> Bool {
        let lines = text
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        guard lines.count >= minimumRepeats else { return false }
        let suffix = Array(lines.suffix(minimumRepeats))
        guard let first = suffix.first else { return false }
        return suffix.allSatisfy { $0 == first && $0.count >= 12 }
    }

    private static func trimRepeatedLines(_ text: String, minimumRepeats: Int) -> String? {
        var lines = text.components(separatedBy: .newlines)
        guard lines.count >= minimumRepeats else { return nil }

        while lines.count >= minimumRepeats {
            let trimmed = lines.suffix(minimumRepeats).map {
                $0.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            guard let first = trimmed.first, !first.isEmpty, first.count >= 12 else { break }
            if trimmed.allSatisfy({ $0 == first }) {
                lines.removeLast()
            } else {
                break
            }
        }

        let result = lines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        return result.isEmpty ? nil : result
    }

}
