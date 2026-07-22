import Foundation
import UIKit
import MLXDeps

@MainActor
@Observable
final class ChatViewModel {
    var messages: [ChatMessage] = []
    var isGenerating = false
    var error: String?
    var statusMessage: String?
    var thinkingEnabled = false
    private var streamingMessageIndex: Int?
    private var pendingToolMessageIndex: Int?
    var generationTask: Task<Void, Never>?

    /// The selected model spec (set by ModelPickerView).
    var selectedModel: ModelSpec? {
        didSet {
            if let spec = selectedModel {
                SettingsManager.shared.loadedModelId = spec.hfId
                SettingsManager.shared.loadedModelName = spec.displayName
            }
        }
    }

    private var engine: MLXEngine? { SettingsManager.shared.sharedEngine }

    var isThinkingModel: Bool {
        guard let model = selectedModel ?? resolvedModel else { return false }
        return model.isThinkingModel && model.supportsNoThink
    }

    /// Resolve current model from SettingsManager if no explicit selection.
    private var resolvedModel: ModelSpec? {
        guard let hfId = SettingsManager.shared.loadedModelId else { return nil }
        return ModelRegistry.find(hfId: hfId)
    }

    private var currentModel: ModelSpec? {
        selectedModel ?? resolvedModel
    }

    func sendMessage(text: String, image: UIImage?) async {
        guard let model = currentModel else {
            error = "No model selected — tap the model picker to choose one"
            return
        }
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || image != nil else { return }

        if image != nil && !model.supportsVision {
            error = "'\(model.displayName)' doesn't support images."
            return
        }

        let settings = SettingsManager.shared
        let maxEdge = CGFloat(settings.maxImageDimension)
        let quality = settings.jpegQuality

        // Resize and compress image
        let processedImage: UIImage? = image.flatMap { img in
            let resized = resizeImage(img, maxEdge: maxEdge)
            return compressImage(resized, quality: quality)
        }

        let userMessage = ChatMessage(role: .user, text: text, image: processedImage)
        messages.append(userMessage)

        isGenerating = true
        error = nil

        do {
            var engine = self.engine
            let needsVision = image != nil
            let hasBraveKey = !settings.braveAPIKey.isEmpty
            let preflightTool = Self.preflightToolCall(
                for: text,
                hasImage: image != nil,
                toolsEnabled: settings.toolsEnabled,
                braveAPIKeyAvailable: hasBraveKey
            )

            // Surfaces download progress while loading so a (re)download — e.g.
            // after iOS evicts cached weights — never looks like a frozen
            // "Loading…". `base` is the label shown before any progress arrives.
            func loadProgress(_ base: String) -> @Sendable (Progress) -> Void {
                { [weak self] progress in
                    let fraction = progress.fractionCompleted
                    Task { @MainActor in
                        guard let self, self.statusMessage != nil else { return }
                        if fraction > 0, fraction < 1 {
                            self.statusMessage =
                                "\(base) \(Int((fraction * 100).rounded()))%"
                        }
                    }
                }
            }

            if engine == nil {
                let base = "Loading \(model.displayName)..."
                statusMessage = base
                let newEngine = MLXEngine(memoryLimitGB: settings.gpuMemoryLimitGB)
                _ = try await newEngine.loadModel(
                    id: model.hfId, forVision: needsVision, progress: loadProgress(base))
                settings.sharedEngine = newEngine
                engine = newEngine
            } else {
                let currentlyVLM = await engine!.loadedAsVLM
                if needsVision && !currentlyVLM {
                    // Currently loaded as LLM but need VLM for image — reload
                    let base = "Loading vision model..."
                    statusMessage = base
                    _ = try await engine!.loadModel(
                        id: model.hfId, forVision: true, progress: loadProgress(base))
                } else if !needsVision && currentlyVLM {
                    // Currently loaded as VLM but no image — reload as LLM for speed + better tools
                    let base = "Loading \(model.displayName)..."
                    statusMessage = base
                    _ = try await engine!.loadModel(
                        id: model.hfId, forVision: false, progress: loadProgress(base))
                }
            }

            guard let engine else {
                error = "Failed to load model"
                isGenerating = false
                return
            }

            if needsVision {
                await engine.clearCache()
            }

            statusMessage = thinkingEnabled ? "Thinking..." : "Generating..."

            var chatMessages: [Chat.Message] = []
            let toolsActive = Self.shouldEnableTools(
                for: text,
                hasImage: image != nil,
                toolsEnabled: settings.toolsEnabled
            )
            let useDirectToolPrefetch = preflightTool != nil
            let currentInfoPrompt = Self.requiresCurrentInfo(text)
            let simplePrompt = Self.isSimplePrompt(text)
            let creativePrompt = Self.isCreativePrompt(text)
            let imageDescriptionPrompt = image != nil && Self.isImageDescriptionPrompt(text)

            // Always add system prompt with variable substitution
            let systemPromptText = settings.processedSystemPrompt()
            var systemText = systemPromptText
            if model.supportsNoThink {
                if thinkingEnabled {
                    systemText += "\nThinking mode is enabled for this turn. Keep reasoning brief — at most a few short sentences — then emit </think> and give the answer."
                    if simplePrompt {
                        systemText += " For greetings and short prompts, one short thought is enough."
                    }
                } else {
                    systemText += "\nThinking mode is disabled for this turn. Reply with only the final answer. Do not output hidden reasoning, self-corrections, or deliberation. For simple questions, answer in one short sentence."
                }
            }
            if toolsActive {
                systemText += "\nYou have tools available but only use them when needed. For simple conversation, respond directly."
                systemText += "\nTo call a tool, reply with ONLY this exact XML format and no trailing text:"
                systemText += "\n<tool_call>\n<function=tool_name>\n<parameter=param_name>value</parameter>\n</function>\n</tool_call>"
                if currentInfoPrompt, hasBraveKey {
                    systemText += "\nThis user is asking for current or recent information. You must call the web_search tool before answering."
                } else if currentInfoPrompt, !hasBraveKey {
                    systemText += "\nThis user is asking for current or recent information, but web search is not configured in this session. Do not claim to have browsed the web."
                }
            }
            if useDirectToolPrefetch {
                systemText += "\nA tool result is already provided in the conversation. Use it if relevant and answer directly. Do not say that you cannot browse or access the web."
            }
            chatMessages.append(.system(systemText))

            let lastIndex = messages.count - 1
            for (idx, msg) in messages.enumerated() {
                switch msg.role {
                case .system:
                    chatMessages.append(.system(msg.text))
                case .user:
                    if idx == lastIndex, model.supportsVision,
                       let uiImage = msg.image, let ciImage = CIImage(image: uiImage) {
                        chatMessages.append(.user(msg.text, images: [.ciImage(ciImage)]))
                    } else {
                        chatMessages.append(.user(msg.text))
                    }
                case .assistant:
                    chatMessages.append(.assistant(msg.text))
                case .tool:
                    chatMessages.append(.tool(msg.text))
                }
            }

            let contextSize = settings.contextSize
            // Tools + thinking tags burn tokens before the actual response
            let maxTokens = imageDescriptionPrompt ? min(360, contextSize / 6) :
                image != nil ? min(280, contextSize / 7) :
                (toolsActive || useDirectToolPrefetch) ? min(400, contextSize / 10) :
                simplePrompt && !thinkingEnabled ? min(48, contextSize / 24) :
                simplePrompt && thinkingEnabled ? min(384, contextSize / 10) :
                creativePrompt ? min(640, contextSize / 6) :
                thinkingEnabled ? min(1200, contextSize / 4) :
                min(160, contextSize / 16)

            // Qwen3.5 official sampling (HF model card): thinking-general
            // temp 1.0 / top_p 0.95; non-thinking-general temp 0.7 / top_p 0.8; both
            // top_k 20 (top_k 40 for creative/reasoning), min_p 0. Repetition is
            // controlled by a non-zero presence_penalty (~1.5) with repetition_penalty
            // left at 1.0 (off) — a change from Qwen3, which used repetition_penalty.
            // We keep a lower temperature for trivial prompts to bias toward short,
            // direct answers, and rely on the `trimRepetition` safety net for the rare
            // loops presence_penalty misses.
            let enableThinking: Bool? = model.isThinkingModel && model.supportsNoThink
                ? thinkingEnabled : nil
            let temperature: Float = thinkingEnabled ? 1.0 : (simplePrompt ? 0.35 : 0.7)
            let topP: Float = thinkingEnabled ? 0.95 : 0.8
            let topK: Int = creativePrompt ? 40 : 20
            let presencePenalty: Float = 1.5

            // Build tools if enabled
            let toolSchemas: [ToolSpec]?
            let toolDispatch: (@Sendable (String, [String: String]) async -> (name: String, result: String))?

            if toolsActive && !useDirectToolPrefetch {
                let braveKey = settings.braveAPIKey
                toolSchemas = ToolRegistry.allSchemas(braveAPIKeyAvailable: hasBraveKey)
                toolDispatch = { @Sendable name, args in
                    await ToolRegistry.dispatchByName(
                        name: name, arguments: args,
                        braveAPIKey: hasBraveKey ? braveKey : nil
                    )
                }
            } else {
                toolSchemas = nil
                toolDispatch = nil
            }

            let onToolCall: (@MainActor @Sendable (String) -> Void)?
            if toolsActive && !useDirectToolPrefetch {
                onToolCall = { [weak self] toolName in
                    guard let self else { return }
                    let displayName = Self.toolDisplayName(toolName)
                    self.statusMessage = displayName
                    self.messages.append(
                        ChatMessage(
                            role: .tool,
                            text: "",
                            displayText: displayName,
                            toolName: toolName
                        )
                    )
                    self.pendingToolMessageIndex = self.messages.count - 1
                }
            } else {
                onToolCall = nil
            }

            let onToolResult: (@MainActor @Sendable (String, String) -> Void)?
            if toolsActive && !useDirectToolPrefetch {
                onToolResult = { [weak self] toolName, result in
                    guard let self else { return }
                    if let idx = self.pendingToolMessageIndex, idx < self.messages.count {
                        self.messages[idx].text = result
                        self.messages[idx].displayText = Self.toolDisplayName(toolName)
                    } else {
                        self.messages.append(
                            ChatMessage(
                                role: .tool,
                                text: result,
                                displayText: Self.toolDisplayName(toolName),
                                toolName: toolName
                            )
                        )
                    }
                    self.pendingToolMessageIndex = nil
                }
            } else {
                onToolResult = nil
            }

            if let preflightTool {
                let displayName = Self.toolDisplayName(preflightTool.name)
                statusMessage = displayName
                messages.append(
                    ChatMessage(
                        role: .tool,
                        text: "",
                        displayText: displayName,
                        toolName: preflightTool.name
                    )
                )
                pendingToolMessageIndex = messages.count - 1

                let preflightResult = await ToolRegistry.dispatchByName(
                    name: preflightTool.name,
                    arguments: preflightTool.arguments,
                    braveAPIKey: hasBraveKey ? settings.braveAPIKey : nil
                )

                if let idx = pendingToolMessageIndex, idx < messages.count {
                    messages[idx].text = preflightResult.result
                    messages[idx].displayText = Self.toolDisplayName(preflightResult.name)
                }
                pendingToolMessageIndex = nil
                chatMessages.append(.tool(preflightResult.result))
            }

            // Set up streaming callback
            let streaming = settings.streamingEnabled
            let onChunk: (@MainActor @Sendable (String) -> Void)?
            if streaming {
                var placeholder = ChatMessage(role: .assistant, text: "")
                placeholder.isThinkingInProgress = thinkingEnabled
                messages.append(placeholder)
                streamingMessageIndex = messages.count - 1
                let revealThinking = thinkingEnabled
                onChunk = { [weak self] chunk in
                    guard let self, let idx = self.streamingMessageIndex,
                          idx < self.messages.count else { return }
                    self.messages[idx].text += chunk
                    let (thinking, answer) = ThinkingText.split(self.messages[idx].text)
                    self.messages[idx].displayText = answer
                    if revealThinking {
                        self.messages[idx].thinkingText = thinking.isEmpty ? nil : thinking
                        self.messages[idx].isThinkingInProgress =
                            ThinkingText.isOpen(self.messages[idx].text)
                    }
                }
            } else {
                onChunk = nil
            }

            let result = try await engine.generateChat(
                messages: chatMessages,
                maxTokens: maxTokens,
                temperature: temperature,
                topP: topP,
                topK: topK,
                presencePenalty: presencePenalty,
                enableThinking: enableThinking,
                hasImage: image != nil,
                tools: toolSchemas,
                toolDispatch: toolDispatch,
                onToolCall: onToolCall,
                onToolResult: onToolResult,
                onChunk: onChunk
            )

            var finalOutput = Self.postProcessAssistantOutput(
                result.cleanedOutput,
                simplePrompt: simplePrompt,
                thinkingEnabled: thinkingEnabled
            )
            var finalMetrics = result.metrics
            var finalThinking = thinkingEnabled ? result.thinkingText : ""

            if thinkingEnabled,
               !finalThinking.isEmpty,
               finalOutput.isEmpty,
               image == nil,
               !toolsActive
            {
                // Main turn used all its tokens inside <think> and never
                // produced an answer. Do a short non-thinking pass so the user
                // still gets a reply.
                let answerSystemText = systemText + "\nReasoning has already been done. Reply with only the user-facing answer in one short sentence. Do not output <think> tags."
                var answerMessages = chatMessages
                answerMessages[0] = .system(answerSystemText)
                let answerResult = try await engine.generateChat(
                    messages: answerMessages,
                    maxTokens: min(64, maxTokens / 4),
                    temperature: 0.3,
                    topP: 0.8,
                    repetitionPenalty: 1.0,
                    enableThinking: false,
                    hasImage: false,
                    tools: nil,
                    toolDispatch: nil,
                    onToolCall: nil,
                    onToolResult: nil,
                    onChunk: nil
                )
                let answerOutput = Self.postProcessAssistantOutput(
                    answerResult.cleanedOutput,
                    simplePrompt: simplePrompt,
                    thinkingEnabled: false
                )
                if !answerOutput.isEmpty {
                    finalOutput = answerOutput
                    finalMetrics = answerResult.metrics
                }
            }

            if !thinkingEnabled,
               image == nil,
               !toolsActive,
               Self.looksLikeReasoningOnlyOutput(finalOutput)
            {
                let retrySystemText = systemText + "\nYour previous draft exposed reasoning instead of an answer. Respond again with only the user-facing answer. No preamble. No bullets. No analysis. Maximum 12 words."
                var retryMessages = chatMessages
                retryMessages[0] = .system(retrySystemText)
                let retryResult = try await engine.generateChat(
                    messages: retryMessages,
                    maxTokens: min(32, maxTokens),
                    temperature: 0.2,
                    topP: 0.7,
                    repetitionPenalty: 1.0,
                    enableThinking: false,
                    hasImage: false,
                    tools: nil,
                    toolDispatch: nil,
                    onToolCall: nil,
                    onToolResult: nil,
                    onChunk: nil
                )
                let retryOutput = Self.postProcessAssistantOutput(
                    retryResult.cleanedOutput,
                    simplePrompt: simplePrompt,
                    thinkingEnabled: false
                )
                if !retryOutput.isEmpty, !Self.looksLikeReasoningOnlyOutput(retryOutput) {
                    finalOutput = retryOutput
                    finalMetrics = retryResult.metrics
                    finalThinking = ""
                }
            }

            if let err = result.error {
                // Remove the streaming placeholder on error
                if streaming, let idx = streamingMessageIndex, idx < messages.count {
                    messages.remove(at: idx)
                }
                removePendingToolPlaceholderIfNeeded()
                error = err
            } else if streaming, let idx = streamingMessageIndex, idx < messages.count {
                let (streamedThinking, streamedAnswer) = ThinkingText.split(messages[idx].text)
                let shouldKeepStreamedOutput =
                    !streamedAnswer.isEmpty &&
                    !Self.looksLikeReasoningOnlyOutput(streamedAnswer) &&
                    streamedAnswer.count > finalOutput.count + 40

                messages[idx].text = shouldKeepStreamedOutput ? streamedAnswer : finalOutput
                messages[idx].displayText = nil
                messages[idx].metrics = finalMetrics
                let thinkingForDisplay = thinkingEnabled
                    ? (shouldKeepStreamedOutput ? streamedThinking : finalThinking)
                    : ""
                messages[idx].thinkingText = thinkingForDisplay.isEmpty ? nil : thinkingForDisplay
                messages[idx].isThinkingInProgress = false
            } else {
                var assistant = ChatMessage(
                    role: .assistant,
                    text: finalOutput,
                    metrics: finalMetrics
                )
                if !finalThinking.isEmpty {
                    assistant.thinkingText = finalThinking
                }
                messages.append(assistant)
            }

            streamingMessageIndex = nil
        } catch is CancellationError {
            // Stopped by user — no error to show
            removePendingToolPlaceholderIfNeeded()
        } catch {
            self.error = error.localizedDescription
            removePendingToolPlaceholderIfNeeded()
        }

        statusMessage = nil
        isGenerating = false
    }

    func stopGeneration() {
        generationTask?.cancel()
        generationTask = nil

        // Finalize any streaming message with whatever text was generated so far
        if let idx = streamingMessageIndex, idx < messages.count {
            let (thinking, cleaned) = ThinkingText.split(messages[idx].text)
            if cleaned.isEmpty && thinking.isEmpty {
                messages.remove(at: idx)
            } else {
                messages[idx].text = cleaned
                messages[idx].displayText = nil
                if thinkingEnabled, !thinking.isEmpty {
                    messages[idx].thinkingText = thinking
                }
                messages[idx].isThinkingInProgress = false
            }
        }
        streamingMessageIndex = nil
        removePendingToolPlaceholderIfNeeded()
        statusMessage = nil
        isGenerating = false
    }

    func clearSession() {
        stopGeneration()
        messages = []
        error = nil
    }

    private static func postProcessAssistantOutput(
        _ text: String,
        simplePrompt: Bool,
        thinkingEnabled: Bool
    ) -> String {
        let cleaned = ThinkingText.stripped(text)
        guard !cleaned.isEmpty, !thinkingEnabled else { return cleaned }

        if let leaked = salvageFromReasoningLeak(cleaned, simplePrompt: simplePrompt) {
            return leaked
        }

        return cleaned
    }

    private static func salvageFromReasoningLeak(_ text: String, simplePrompt: Bool) -> String? {
        let lines = text
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        let reasoningKeywords = [
            "thinking process", "analyze the", "analyze the request", "analyze the input",
            "constraint:", "decision:", "re-evaluating", "let's ", "however", "since ",
            "wait", "actually", "i should", "i can", "i need to", "for simple questions"
        ]

        let reasoningLineCount = lines.filter { line in
            let lower = line.lowercased()
            return lower.hasPrefix("*") ||
                lower.hasPrefix("-") ||
                lower.contains("constraint:") ||
                reasoningKeywords.contains(where: { lower.contains($0) })
        }.count

        let looksLikeReasoningLeak = reasoningLineCount >= 2 ||
            reasoningKeywords.contains(where: { text.lowercased().contains($0) }) ||
            text.contains("Constraint:")
        guard looksLikeReasoningLeak else { return nil }

        if let quoted = lastQuotedSentence(in: text) {
            return quoted
        }

        let candidateLines = lines.filter { line in
            let lower = line.lowercased()
            guard !lower.hasPrefix("*"), !lower.hasPrefix("-") else { return false }
            guard !reasoningKeywords.contains(where: { lower.contains($0) }) else { return false }
            guard !lower.contains("constraint:"), !lower.contains("decision:") else { return false }
            return line.count <= (simplePrompt ? 90 : 180)
        }

        return candidateLines.last
    }

    private static func looksLikeReasoningOnlyOutput(_ text: String) -> Bool {
        let normalized = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return true }
        let markers = [
            "thinking process", "analyze the request", "analyze the input",
            "constraint:", "decision:", "re-evaluating", "let's ",
            "i should", "i need to", "for simple questions"
        ]
        if markers.contains(where: { normalized.contains($0) }) {
            return true
        }

        let lines = normalized
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let bulletLikeLines = lines.filter {
            $0.hasPrefix("1.") || $0.hasPrefix("2.") || $0.hasPrefix("*") || $0.hasPrefix("-")
        }
        let analyticalBulletLines = bulletLikeLines.filter { line in
            markers.contains(where: { line.contains($0) })
        }
        if !analyticalBulletLines.isEmpty && analyticalBulletLines.count * 2 >= max(1, bulletLikeLines.count) {
            return true
        }
        return false
    }

    private static func lastQuotedSentence(in text: String) -> String? {
        let pattern = #""([^"\n]{2,160}[.!?])""#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let matches = regex.matches(in: text, range: NSRange(text.startIndex..., in: text))
        for match in matches.reversed() {
            guard let range = Range(match.range(at: 1), in: text) else { continue }
            let candidate = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
            if !candidate.isEmpty {
                return candidate
            }
        }
        return nil
    }

    private func removePendingToolPlaceholderIfNeeded() {
        if let idx = pendingToolMessageIndex, idx < messages.count, messages[idx].text.isEmpty {
            messages.remove(at: idx)
        }
        pendingToolMessageIndex = nil
    }

    private static func shouldEnableTools(for text: String, hasImage: Bool, toolsEnabled: Bool) -> Bool {
        guard toolsEnabled, !hasImage else { return false }

        let normalized = text.lowercased()
        if normalized.contains("http://") || normalized.contains("https://") || normalized.contains("www.") {
            return true
        }

        let triggers = [
            "latest", "current", "today", "recent", "news", "headline", "weather", "forecast",
            "search", "look up", "lookup", "find online", "browse", "website", "web page",
            "article", "url", "link", "stock price", "price of", "score", "exchange rate"
        ]
        return triggers.contains { normalized.contains($0) }
    }

    private static func requiresCurrentInfo(_ text: String) -> Bool {
        let normalized = text.lowercased()
        let triggers = [
            "latest", "current", "today", "recent", "news", "headline",
            "just happened", "what's happening", "what is happening", "update"
        ]
        return triggers.contains { normalized.contains($0) }
    }

    private struct PreflightToolCall {
        let name: String
        let arguments: [String: String]
    }

    private static func preflightToolCall(
        for text: String,
        hasImage: Bool,
        toolsEnabled: Bool,
        braveAPIKeyAvailable: Bool
    ) -> PreflightToolCall? {
        guard shouldEnableTools(for: text, hasImage: hasImage, toolsEnabled: toolsEnabled) else {
            return nil
        }

        if let url = firstURL(in: text) {
            return PreflightToolCall(name: "url_fetch", arguments: ["url": url])
        }

        if braveAPIKeyAvailable, requiresCurrentInfo(text) {
            return PreflightToolCall(name: "web_search", arguments: ["query": text])
        }

        return nil
    }

    private static func firstURL(in text: String) -> String? {
        guard let detector = try? NSDataDetector(types: NSTextCheckingResult.CheckingType.link.rawValue) else {
            return nil
        }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = detector.matches(in: text, options: [], range: range)
        for match in matches {
            guard let url = match.url?.absoluteString else { continue }
            if url.hasPrefix("http://") || url.hasPrefix("https://") {
                return url
            }
        }
        return nil
    }

    private static func isSimplePrompt(_ text: String) -> Bool {
        let normalized = text
            .lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if normalized.count <= 24 {
            let simplePhrases = [
                "hi", "hello", "hey", "yo", "sup", "thanks", "thank you",
                "ok", "okay", "cool", "nice", "great", "morning", "good morning",
                "afternoon", "good afternoon", "evening", "good evening"
            ]
            if simplePhrases.contains(normalized) {
                return true
            }
        }
        return normalized.count <= 40 &&
            !normalized.contains("?") &&
            !normalized.contains("http://") &&
            !normalized.contains("https://")
    }

    private static func isCreativePrompt(_ text: String) -> Bool {
        let normalized = text
            .lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let triggers = [
            "write a story", "tell me a story", "story about", "short story",
            "write me", "write an", "write about", "fiction", "fairy tale",
            "bedtime story", "poem", "haiku", "sonnet", "scene", "chapter"
        ]

        return triggers.contains { normalized.contains($0) }
    }

    private static func isImageDescriptionPrompt(_ text: String) -> Bool {
        let normalized = text
            .lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)

        let triggers = [
            "describe the image", "describe this image", "what's in this image",
            "what is in this image", "what's in the picture", "what is in the picture",
            "describe the picture", "analyze this image", "analyse this image",
            "caption this image"
        ]

        return triggers.contains { normalized.contains($0) }
    }


    private static func toolDisplayName(_ name: String) -> String {
        switch name {
        case "web_search": return "Searching the web..."
        case "url_fetch": return "Fetching URL..."
        default: return "Using \(name)..."
        }
    }

    private func resizeImage(_ image: UIImage, maxEdge: CGFloat) -> UIImage {
        let size = image.size
        let longestEdge = max(size.width, size.height)
        guard longestEdge > maxEdge else { return image }

        let scale = maxEdge / longestEdge
        let newSize = CGSize(width: size.width * scale, height: size.height * scale)

        let renderer = UIGraphicsImageRenderer(size: newSize)
        return renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: newSize))
        }
    }

    private func compressImage(_ image: UIImage, quality: Double) -> UIImage? {
        guard let data = image.jpegData(compressionQuality: CGFloat(quality)) else { return image }
        return UIImage(data: data) ?? image
    }
}
