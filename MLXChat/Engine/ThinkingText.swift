import Foundation

/// Utilities for separating Qwen-style `<think>…</think>` blocks from the
/// user-facing answer. Used both during streaming (where the closing tag may
/// not have arrived yet) and on the final response.
enum ThinkingText {

    /// Split a model response into a thinking trace and the user-facing answer.
    ///
    /// Handles three streaming states:
    /// - both tags present: thinking content is between them, answer is what's
    ///   outside the block
    /// - `<think>` open but no `</think>` yet: still mid-reasoning, everything
    ///   after the open tag is in-progress thinking
    /// - `</think>` only (model entered thinking before the first chunk arrived
    ///   and the open tag was dropped by the template): everything before the
    ///   close tag is thinking
    static func split(_ text: String) -> (thinking: String, answer: String) {
        if let openRange = text.range(of: "<think>") {
            let afterOpen = openRange.upperBound
            if let closeRange = text.range(of: "</think>", range: afterOpen..<text.endIndex) {
                let thinking = String(text[afterOpen..<closeRange.lowerBound])
                let answer =
                    String(text[..<openRange.lowerBound]) +
                    String(text[closeRange.upperBound...])
                return (
                    thinking.trimmingCharacters(in: .whitespacesAndNewlines),
                    answer.trimmingCharacters(in: .whitespacesAndNewlines)
                )
            }
            let thinking = String(text[afterOpen...])
            let answer = String(text[..<openRange.lowerBound])
            return (
                thinking.trimmingCharacters(in: .whitespacesAndNewlines),
                answer.trimmingCharacters(in: .whitespacesAndNewlines)
            )
        }

        if let closeRange = text.range(of: "</think>") {
            let thinking = String(text[..<closeRange.lowerBound])
            let answer = String(text[closeRange.upperBound...])
            return (
                thinking.trimmingCharacters(in: .whitespacesAndNewlines),
                answer.trimmingCharacters(in: .whitespacesAndNewlines)
            )
        }

        return ("", text.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// Strip `<think>` blocks entirely. Used when the caller doesn't care about
    /// the thinking content (e.g. building final assistant text for the
    /// conversation history that is replayed to the model).
    static func stripped(_ text: String) -> String {
        split(text).answer
    }

    /// True if `<think>` is open but `</think>` has not yet arrived.
    static func isOpen(_ text: String) -> Bool {
        guard text.contains("<think>") else { return false }
        return !text.contains("</think>")
    }
}
