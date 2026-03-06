import Foundation
import Hub
import MLX
import MLXLMCommon
import MLXLLM
import MLXVLM
import Tokenizers

actor MLXEngine {

    private var modelContainer: ModelContainer?
    private var currentModelId: String?
    private var isVLM = false

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

        let config = ModelConfiguration(
            id: id,
            // Qwen3.5 tool use maps to the qwen3_coder-style XML function format.
            toolCallFormat: Self.toolCallFormat(for: id)
        )
        let progressHandler: @Sendable (Progress) -> Void = progress ?? { _ in }

        if forVision {
            // Vision requests require VLMModelFactory
            modelContainer = try await VLMModelFactory.shared.loadContainer(
                configuration: config, progressHandler: progressHandler)
            isVLM = true
        } else {
            // Text-only: try LLM first (2x faster, better tool support), fall back to VLM
            do {
                modelContainer = try await LLMModelFactory.shared.loadContainer(
                    configuration: config, progressHandler: progressHandler)
                isVLM = false
            } catch {
                modelContainer = try await VLMModelFactory.shared.loadContainer(
                    configuration: config, progressHandler: progressHandler)
                isVLM = true
            }
        }

        currentModelId = id
        return CFAbsoluteTimeGetCurrent() - start
    }

    var loadedAsVLM: Bool { isVLM }

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
        repetitionPenalty: Float? = nil,
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

        // Aggressively clear cache, disable it entirely for vision
        Memory.cacheLimit = 0
        Memory.clearCache()
        if !hasImage {
            Memory.cacheLimit = 512 * 1024 * 1024
        }
        let baselineMemory = Double(Memory.activeMemory) / (1024 * 1024)
        Memory.peakMemory = 0

        var additionalCtx: [String: any Sendable]?
        if let enableThinking {
            additionalCtx = ["enable_thinking": enableThinking]
        }

        var currentMessages = messages

        var generateParams = GenerateParameters(
            temperature: temperature,
            topP: topP
        )
        if let repetitionPenalty {
            generateParams.repetitionPenalty = repetitionPenalty
            generateParams.repetitionContextSize = 64
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
                            if chunkCount >= values.maxTokens
                                || (generatedText.contains("<think>") && !generatedText.contains("</think>") && generatedText.count >= min(values.maxTokens, 160))
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
        let cleaned = Self.stripThinkingTags(finalOutput)
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
        messages: [Chat.Message],
        tools: [ToolSpec]?,
        additionalContext: [String: any Sendable]?,
        hasImage: Bool
    ) async throws -> LMInput {
        if !hasImage,
           let chatTemplate = correctedChatTemplateIfNeeded(
               configuration: context.configuration,
               additionalContext: additionalContext
           ) {
            let rawMessages = DefaultMessageGenerator().generate(messages: messages)
            let promptTokens = try context.tokenizer.applyChatTemplate(
                messages: rawMessages,
                chatTemplate: .literal(chatTemplate),
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

    private static func correctedChatTemplateIfNeeded(
        configuration: ModelConfiguration,
        additionalContext: [String: any Sendable]?
    ) -> String? {
        guard
            isQwen35_4B(configuration.name),
            let enableThinking = additionalContext?["enable_thinking"] as? Bool,
            enableThinking == false,
            let chatTemplate = loadChatTemplate(for: configuration)
        else {
            return nil
        }

        return patchQwen35_4BNonThinkingTemplate(chatTemplate)
    }

    private static func loadChatTemplate(for configuration: ModelConfiguration) -> String? {
        let modelDirectory = configuration.modelDirectory(hub: HubApi())
        let templateURL = modelDirectory.appending(path: "chat_template.jinja")
        return try? String(contentsOf: templateURL, encoding: .utf8)
    }

    private static func patchQwen35_4BNonThinkingTemplate(_ chatTemplate: String) -> String {
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

    private static func isQwen35_4B(_ name: String) -> Bool {
        let normalized = name.lowercased()
        return (normalized.contains("qwen3.5") || normalized.contains("qwen3_5"))
            && normalized.contains("4b")
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

    private static func stripThinkingTags(_ text: String) -> String {
        var result = text
        // Strip matched <think>...</think> pairs
        let pattern = #"<think>[\s\S]*?</think>"#
        if let regex = try? NSRegularExpression(pattern: pattern, options: []) {
            let range = NSRange(result.startIndex..., in: result)
            result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: "")
        }
        // Strip orphaned </think> — only discard content before it if there's
        // substantial content after it (otherwise it's the actual response)
        if let closeRange = result.range(of: "</think>") {
            let after = String(result[closeRange.upperBound...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if after.count >= 2 {
                result = after
            } else {
                result = result.replacingOccurrences(of: "</think>", with: "")
            }
        }
        // Strip orphaned <think> — only discard content after it if there's
        // substantial content before it
        if let openRange = result.range(of: "<think>") {
            let before = String(result[..<openRange.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if before.count >= 2 {
                result = before
            } else {
                result = result.replacingOccurrences(of: "<think>", with: "")
            }
        }
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
