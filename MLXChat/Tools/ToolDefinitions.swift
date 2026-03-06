import Foundation
import MLXLMCommon
import Tokenizers

struct ToolRegistry {

    static func allSchemas(braveAPIKeyAvailable: Bool) -> [ToolSpec] {
        var schemas: [ToolSpec] = [
            urlFetchSchema,
        ]
        if braveAPIKeyAvailable {
            schemas.insert(webSearchSchema, at: 0)
        }
        return schemas
    }

    static func dispatch(
        toolCall: ToolCall,
        braveAPIKey: String?
    ) async -> (name: String, result: String) {
        let name = toolCall.function.name
        var args: [String: String] = [:]
        for (k, v) in toolCall.function.arguments {
            args[k] = v.stringValue ?? "\(v)"
        }
        return await dispatchByName(name: name, arguments: args, braveAPIKey: braveAPIKey)
    }

    static func dispatchByName(
        name: String,
        arguments: [String: String],
        braveAPIKey: String?
    ) async -> (name: String, result: String) {
        let result: String
        switch name {
        case "web_search":
            let query = arguments["query"] ?? ""
            guard let key = braveAPIKey, !key.isEmpty else {
                return (name, "Error: Brave Search API key not configured")
            }
            result = await BraveSearchService.search(query: query, apiKey: key)

        case "url_fetch":
            let url = arguments["url"] ?? ""
            result = await WebFetchService.fetch(urlString: url)

        default:
            result = "Unknown tool: \(name)"
        }
        return (name, result)
    }

    // MARK: - Schemas

    static let webSearchSchema: ToolSpec = [
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

    static let urlFetchSchema: ToolSpec = [
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
}

// MARK: - JSONValue helpers

extension JSONValue {
    var stringValue: String? {
        switch self {
        case .string(let s): return s
        default: return nil
        }
    }

    var doubleValue: Double? {
        switch self {
        case .double(let d): return d
        case .int(let i): return Double(i)
        default: return nil
        }
    }
}
