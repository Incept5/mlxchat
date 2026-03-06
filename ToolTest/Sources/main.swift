import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import MLXVLM
import Tokenizers

// MARK: - Configuration

let defaultModelId = "mlx-community/Qwen3.5-4B-MLX-4bit"

// System prompt — exact same as iOS app
let systemPromptTemplate = """
    You are a helpful assistant. Today is {today} and the time is {time} in the {timezone} timezone. \
    Use British English spelling. Do not use this information unless relevant to the user's question.
    You have tools available but only use them when needed. For simple conversation, respond directly. \
    To call a tool use: <function=tool_name><parameter=param_name>value</parameter></function>. \
    Do not explain your tool choice.
    """

func processedSystemPrompt() -> String {
    var result = systemPromptTemplate
    let f1 = DateFormatter()
    f1.dateFormat = "EEEE, MMMM d, yyyy"
    result = result.replacingOccurrences(of: "{today}", with: f1.string(from: Date()))
    let f2 = DateFormatter()
    f2.dateFormat = "h:mm a"
    result = result.replacingOccurrences(of: "{time}", with: f2.string(from: Date()))
    result = result.replacingOccurrences(of: "{timezone}", with: TimeZone.current.identifier)
    return result
}

// MARK: - Tool Schemas (exact same as iOS app)

let webSearchSchema: ToolSpec = [
    "type": "function",
    "function": [
        "name": "web_search",
        "description": "Search the web for current information. Use this when the user asks about recent events, news, or anything that requires up-to-date information.",
        "parameters": [
            "type": "object",
            "properties": [
                "query": [
                    "type": "string",
                    "description": "The search query",
                ] as [String: any Sendable],
            ] as [String: any Sendable],
            "required": ["query"],
        ] as [String: any Sendable],
    ] as [String: any Sendable],
]

let urlFetchSchema: ToolSpec = [
    "type": "function",
    "function": [
        "name": "url_fetch",
        "description": "Fetch and read the text content of a web page URL. Use this when the user provides a URL or when you need to read a specific web page.",
        "parameters": [
            "type": "object",
            "properties": [
                "url": [
                    "type": "string",
                    "description": "The URL to fetch",
                ] as [String: any Sendable],
            ] as [String: any Sendable],
            "required": ["url"],
        ] as [String: any Sendable],
    ] as [String: any Sendable],
]

func allToolSchemas() -> [ToolSpec] {
    [webSearchSchema, urlFetchSchema]
}

// MARK: - Tool dispatch (mock)

func dispatchTool(name: String, arguments: [String: String]) async -> String {
    switch name {
    case "web_search":
        return "[Web search results for: \(arguments["query"] ?? "?")] — 1. Result one 2. Result two 3. Result three"
    case "url_fetch":
        return "[Fetched content from: \(arguments["url"] ?? "?")]"
    default:
        return "Unknown tool: \(name)"
    }
}

// MARK: - Helpers

func stripThinkingTags(_ text: String) -> String {
    var result = text
    let pattern = #"<think>[\s\S]*?</think>"#
    if let regex = try? NSRegularExpression(pattern: pattern, options: []) {
        let range = NSRange(result.startIndex..., in: result)
        result = regex.stringByReplacingMatches(in: result, range: range, withTemplate: "")
    }
    if let closeRange = result.range(of: "</think>") {
        let after = String(result[closeRange.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines)
        if after.count >= 2 {
            result = after
        } else {
            result = result.replacingOccurrences(of: "</think>", with: "")
        }
    }
    if let openRange = result.range(of: "<think>") {
        let before = String(result[..<openRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        if before.count >= 2 {
            result = before
        } else {
            result = result.replacingOccurrences(of: "<think>", with: "")
        }
    }
    return result.trimmingCharacters(in: .whitespacesAndNewlines)
}

struct ParsedToolCall: CustomStringConvertible {
    let name: String
    let arguments: [String: String]
    var description: String { "\(name)(\(arguments))" }
}

func parseToolCall(from text: String) -> ParsedToolCall? {
    // Pattern 1: Full format
    let qwenPattern = #"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>"#
    if let regex = try? NSRegularExpression(pattern: qwenPattern, options: [.dotMatchesLineSeparators]) {
        let range = NSRange(text.startIndex..., in: text)
        if let match = regex.firstMatch(in: text, range: range),
           let nameRange = Range(match.range(at: 1), in: text),
           let bodyRange = Range(match.range(at: 2), in: text) {
            return ParsedToolCall(name: String(text[nameRange]).trimmingCharacters(in: .whitespacesAndNewlines),
                                  arguments: parseXMLParameters(String(text[bodyRange])))
        }
    }

    // Pattern 2: Without wrapper
    let xmlFuncPattern = #"<function=([^>]+)>(.*?)</function>"#
    if let regex = try? NSRegularExpression(pattern: xmlFuncPattern, options: [.dotMatchesLineSeparators]) {
        let range = NSRange(text.startIndex..., in: text)
        if let match = regex.firstMatch(in: text, range: range),
           let nameRange = Range(match.range(at: 1), in: text),
           let bodyRange = Range(match.range(at: 2), in: text) {
            return ParsedToolCall(name: String(text[nameRange]).trimmingCharacters(in: .whitespacesAndNewlines),
                                  arguments: parseXMLParameters(String(text[bodyRange])))
        }
    }

    // Pattern 3: Loose — no </function>
    let loosePattern = #"<tool_call>\s*<function=([a-z_]+)\s*>?\s*[\s\S]*?</tool_call>"#
    if let regex = try? NSRegularExpression(pattern: loosePattern, options: [.dotMatchesLineSeparators]) {
        let range = NSRange(text.startIndex..., in: text)
        if let match = regex.firstMatch(in: text, range: range),
           let nameRange = Range(match.range(at: 1), in: text) {
            let name = String(text[nameRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            let fullRange = Range(match.range, in: text)!
            let body = String(text[fullRange])
            return ParsedToolCall(name: name, arguments: parseXMLParameters(body))
        }
    }

    // Pattern 4: Bare <function=name>
    let barePattern = #"<function=([a-z_]+)\s*/?>"#
    if let regex = try? NSRegularExpression(pattern: barePattern, options: []) {
        let range = NSRange(text.startIndex..., in: text)
        if let match = regex.firstMatch(in: text, range: range),
           let nameRange = Range(match.range(at: 1), in: text) {
            return ParsedToolCall(name: String(text[nameRange]).trimmingCharacters(in: .whitespacesAndNewlines),
                                  arguments: parseXMLParameters(text))
        }
    }

    // Pattern 5: JSON in <tool_call> tags
    let jsonPattern = #"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>"#
    if let regex = try? NSRegularExpression(pattern: jsonPattern, options: []) {
        let range = NSRange(text.startIndex..., in: text)
        if let match = regex.firstMatch(in: text, range: range),
           let jsonRange = Range(match.range(at: 1), in: text) {
            let jsonStr = String(text[jsonRange])
            if let data = jsonStr.data(using: .utf8),
               let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let name = dict["name"] as? String {
                let argsDict = dict["arguments"] as? [String: Any] ?? [:]
                var args: [String: String] = [:]
                for (k, v) in argsDict { args[k] = "\(v)" }
                return ParsedToolCall(name: name, arguments: args)
            }
        }
    }

    return nil
}

func parseXMLParameters(_ body: String) -> [String: String] {
    var args: [String: String] = [:]
    let paramPattern = #"<parameter=([^>]+)>([\s\S]*?)</parameter>"#
    if let paramRegex = try? NSRegularExpression(pattern: paramPattern, options: [.dotMatchesLineSeparators]) {
        let matches = paramRegex.matches(in: body, range: NSRange(body.startIndex..., in: body))
        for pm in matches {
            if let keyRange = Range(pm.range(at: 1), in: body),
               let valRange = Range(pm.range(at: 2), in: body) {
                args[String(body[keyRange]).trimmingCharacters(in: .whitespacesAndNewlines)] =
                    String(body[valRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
    }
    return args
}

extension JSONValue {
    var stringValue: String? {
        switch self {
        case .string(let s): return s
        default: return nil
        }
    }
}

func hasRepetition(_ text: String) -> Bool {
    let checkLen = min(text.count, 600)
    guard checkLen >= 80 else { return false }
    let tail = String(text.suffix(checkLen))
    for patternLen in stride(from: 20, through: min(200, checkLen / 3), by: 5) {
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

// MARK: - Generation

func generate(
    container: ModelContainer,
    messages: [Chat.Message],
    maxTokens: Int = 1000,
    temperature: Float = 0.7,
    topP: Float = 0.8,
    repetitionPenalty: Float? = 1.5,
    enableThinking: Bool? = false,
    tools: [ToolSpec]? = nil
) async throws -> (text: String, tokensPerSecond: Double, info: GenerateCompletionInfo?) {
    var additionalCtx: [String: any Sendable]?
    if let enableThinking {
        additionalCtx = ["enable_thinking": enableThinking ? "true" : "false"]
    }

    var generateParams = GenerateParameters(temperature: temperature, topP: topP)
    if let repetitionPenalty {
        generateParams.repetitionPenalty = repetitionPenalty
        generateParams.repetitionContextSize = 64
    }

    nonisolated(unsafe) let userInput = UserInput(
        chat: messages,
        tools: tools,
        additionalContext: additionalCtx
    )

    let input = try await container.prepare(input: userInput)
    let stream = try await container.generate(input: input, parameters: generateParams)

    var outputText = ""
    var completionInfo: GenerateCompletionInfo?
    var parsedToolCall: ToolCall?

    for await generation in stream {
        switch generation {
        case .chunk(let chunk):
            guard parsedToolCall == nil else { continue }
            outputText += chunk
            if outputText.count >= maxTokens { break }
        case .info(let info):
            completionInfo = info
        case .toolCall(let toolCall):
            parsedToolCall = toolCall
        }
    }

    if let parsedToolCall {
        var text = "<function=\(parsedToolCall.function.name)>"
        for (k, v) in parsedToolCall.function.arguments {
            let value = v.stringValue ?? "\(v.anyValue)"
            text += "<parameter=\(k)>\(value)</parameter>"
        }
        text += "</function>"
        return (text, completionInfo?.tokensPerSecond ?? 0, completionInfo)
    }

    return (outputText, completionInfo?.tokensPerSecond ?? 0, completionInfo)
}

// MARK: - Printing

func printHeader(_ title: String) {
    print("\n" + String(repeating: "=", count: 70))
    print("  \(title)")
    print(String(repeating: "=", count: 70))
}

func printOutput(_ raw: String) {
    let cleaned = stripThinkingTags(raw)
    print("  RAW (\(raw.count) chars):")
    for line in raw.prefix(800).split(separator: "\n", omittingEmptySubsequences: false) {
        print("    | \(line)")
    }
    if raw.count > 800 { print("    ...[truncated]") }
    if cleaned != raw && !cleaned.isEmpty {
        print("  CLEANED:")
        for line in cleaned.prefix(400).split(separator: "\n", omittingEmptySubsequences: false) {
            print("    > \(line)")
        }
    }
}

func clearCache() {
    Memory.cacheLimit = 0
    Memory.clearCache()
    Memory.cacheLimit = 512 * 1024 * 1024
}

// MARK: - Main

@main
struct ToolTestApp {
    static func main() async throws {
        print("MLX Tool Test CLI")
        print("Model: \(defaultModelId)")

        let systemPrompt = processedSystemPrompt()
        print("\nSystem prompt:")
        print("  \(systemPrompt)\n")

        let tools = allToolSchemas()
        print("Tools: \(tools.count) (web_search, url_fetch)")

        // Load model with LLMModelFactory (same as iOS text-only path)
        printHeader("Loading model with LLMModelFactory")
        let config = ModelConfiguration(id: defaultModelId, toolCallFormat: .xmlFunction)
        let container: ModelContainer
        do {
            container = try await LLMModelFactory.shared.loadContainer(
                configuration: config) { p in
                    let pct = Int(p.fractionCompleted * 100)
                    if pct % 25 == 0 { print("  \(pct)%") }
                }
            print("  Loaded successfully")
        } catch {
            print("  LLM failed: \(error), trying VLM...")
            container = try await VLMModelFactory.shared.loadContainer(configuration: config)
        }
        clearCache()

        // ── Test 1: Simple greeting (should NOT call tools) ──
        printHeader("Test 1: \"Hi\" — should respond directly, no tools")
        let t1 = try await generate(
            container: container,
            messages: [.system(systemPrompt), .user("Hi")],
            maxTokens: 80,
            tools: tools)
        printOutput(t1.text)
        let t1call = parseToolCall(from: t1.text)
        print("  Tool call: \(t1call?.description ?? "NONE")")
        print("  Tok/s: \(String(format: "%.1f", t1.tokensPerSecond))")
        if let info = t1.info {
            print("  Gen tokens: \(info.generationTokenCount), Gen time: \(String(format: "%.2fs", info.generateTime))")
            print("  Prompt tokens: \(info.promptTokenCount), Prompt time: \(String(format: "%.2fs", info.promptTime))")
        }
        print("  Result: \(t1call == nil ? "PASS — no tool call" : "FAIL — should not call a tool")")
        clearCache()

        // ── Test 2: "What's the time?" (should respond from system prompt, no tools) ──
        printHeader("Test 2: \"What's the time?\" — time is in system prompt, no tool needed")
        let t2 = try await generate(
            container: container,
            messages: [.system(systemPrompt), .user("What's the time?")],
            maxTokens: 80,
            tools: tools)
        printOutput(t2.text)
        let t2call = parseToolCall(from: t2.text)
        print("  Tool call: \(t2call?.description ?? "NONE")")
        print("  Tok/s: \(String(format: "%.1f", t2.tokensPerSecond))")
        print("  Result: \(t2call == nil ? "PASS — used system prompt info" : "FAIL — should not need a tool")")
        clearCache()

        // ── Test 3: "What's the latest news about AI?" (SHOULD call web_search) ──
        printHeader("Test 3: \"What's the latest news about AI?\" — should call web_search")
        let t3 = try await generate(
            container: container,
            messages: [.system(systemPrompt), .user("What's the latest news about AI?")],
            maxTokens: 160,
            tools: tools)
        printOutput(t3.text)
        let t3call = parseToolCall(from: t3.text)
        print("  Tool call: \(t3call?.description ?? "NONE")")
        print("  Tok/s: \(String(format: "%.1f", t3.tokensPerSecond))")
        print("  Result: \(t3call?.name == "web_search" ? "PASS" : "FAIL — expected web_search")")
        clearCache()

        // ── Test 4: Same but WITHOUT tools — baseline ──
        printHeader("Test 4: \"Hi\" — NO tools (baseline speed)")
        let t4 = try await generate(
            container: container,
            messages: [.system(systemPrompt), .user("Hi")],
            maxTokens: 80,
            tools: nil)
        printOutput(t4.text)
        print("  Tok/s: \(String(format: "%.1f", t4.tokensPerSecond))")
        if let info = t4.info {
            print("  Gen tokens: \(info.generationTokenCount), Gen time: \(String(format: "%.2fs", info.generateTime))")
        }
        clearCache()

        // ── Test 5: Multi-turn with tool dispatch ──
        printHeader("Test 5: Multi-turn — \"Search for MLX framework\" with tool dispatch")
        var msgs: [Chat.Message] = [.system(systemPrompt), .user("Search for MLX framework")]
        var toolCallCount = 0

        for turn in 0..<5 {
            let r = try await generate(container: container, messages: msgs, maxTokens: 160, tools: tools)
            print("  [Turn \(turn)] Generated \(r.text.count) chars")
            let call = parseToolCall(from: r.text)

            if let call {
                print("  [Turn \(turn)] TOOL CALL: \(call)")
                toolCallCount += 1
                msgs.append(.assistant(r.text))
                let result = await dispatchTool(name: call.name, arguments: call.arguments)
                msgs.append(.tool(result))
                print("  [Turn \(turn)] Tool result: \(result.prefix(200))")
                clearCache()
                continue
            }

            // Final response
            let cleaned = stripThinkingTags(r.text)
            print("  [Turn \(turn)] FINAL RESPONSE:")
            for line in cleaned.prefix(400).split(separator: "\n", omittingEmptySubsequences: false) {
                print("    > \(line)")
            }
            print("  Tok/s: \(String(format: "%.1f", r.tokensPerSecond))")
            break
        }
        print("  Tool calls made: \(toolCallCount)")
        print("  Result: \(toolCallCount > 0 ? "PASS" : "FAIL — expected at least 1 tool call")")

        printHeader("DONE")
    }
}
