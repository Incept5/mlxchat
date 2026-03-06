import Foundation

enum WebFetchService {

    static func fetch(urlString: String, maxLength: Int = 3000) async -> String {
        guard let url = URL(string: urlString) else {
            return "Error: Invalid URL '\(urlString)'"
        }

        var request = URLRequest(url: url)
        request.timeoutInterval = 15
        request.setValue(
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15",
            forHTTPHeaderField: "User-Agent"
        )

        do {
            let (data, response) = try await URLSession.shared.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                return "Error: Invalid response"
            }
            guard httpResponse.statusCode == 200 else {
                return "Error: HTTP \(httpResponse.statusCode)"
            }

            guard let html = String(data: data, encoding: .utf8) else {
                return "Error: Could not decode response"
            }

            let text = stripHTML(html)
            if text.count > maxLength {
                return String(text.prefix(maxLength)) + "\n\n[Content truncated]"
            }
            return text
        } catch {
            return "Error: \(error.localizedDescription)"
        }
    }

    private static func stripHTML(_ html: String) -> String {
        var text = html

        // Remove script and style blocks
        let blockPatterns = [
            #"<script[^>]*>[\s\S]*?</script>"#,
            #"<style[^>]*>[\s\S]*?</style>"#,
            #"<nav[^>]*>[\s\S]*?</nav>"#,
            #"<header[^>]*>[\s\S]*?</header>"#,
            #"<footer[^>]*>[\s\S]*?</footer>"#,
        ]
        for pattern in blockPatterns {
            if let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) {
                text = regex.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: "")
            }
        }

        // Replace common block elements with newlines
        if let brRegex = try? NSRegularExpression(pattern: #"<br\s*/?>"#, options: .caseInsensitive) {
            text = brRegex.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: "\n")
        }
        if let blockRegex = try? NSRegularExpression(pattern: #"</(p|div|h[1-6]|li|tr)>"#, options: .caseInsensitive) {
            text = blockRegex.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: "\n")
        }

        // Strip all remaining tags
        if let tagRegex = try? NSRegularExpression(pattern: #"<[^>]+>"#) {
            text = tagRegex.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: "")
        }

        // Decode common HTML entities
        text = text
            .replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&#39;", with: "'")
            .replacingOccurrences(of: "&nbsp;", with: " ")

        // Collapse whitespace
        let lines = text.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        return lines.joined(separator: "\n")
    }
}
