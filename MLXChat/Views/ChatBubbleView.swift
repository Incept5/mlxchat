import SwiftUI

struct ChatBubbleView: View {
    let message: ChatMessage
    @State private var showCopied = false

    var body: some View {
        if message.role == .tool {
            toolBubble
        } else {
            chatBubble
        }
    }

    private var chatBubble: some View {
        VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 2) {
            HStack {
                if message.role == .user { Spacer(minLength: 60) }

                VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                    // Copy button for assistant messages
                    if message.role == .assistant && !visibleText.isEmpty {
                        HStack {
                            Spacer()
                            Button {
                                UIPasteboard.general.string = visibleText
                                showCopied = true
                                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                                    showCopied = false
                                }
                            } label: {
                                Image(systemName: showCopied ? "checkmark" : "doc.on.doc")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                        }
                    }

                    if let image = message.image {
                        Image(uiImage: image)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: 200, maxHeight: 200)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }

                    if !(visibleText.isEmpty) {
                        messageTextView
                    }
                }
                .padding(10)
                .background(message.role == .user ? Color.blue : Color(.secondarySystemBackground))
                .clipShape(RoundedRectangle(cornerRadius: 12))

                if message.role == .assistant { Spacer(minLength: 60) }
            }

            // Stats below the bubble
            if message.role == .assistant, let metrics = message.metrics, metrics.tokensPerSecond > 0 {
                HStack(spacing: 8) {
                    Text(String(format: "%.1f tok/s model", metrics.tokensPerSecond))
                    Text("\(metrics.generationTokenCount) tokens")
                    Text(String(format: "%.1fs gen", metrics.generateTimeSeconds))
                    if metrics.totalTimeSeconds > metrics.generateTimeSeconds + 0.2 {
                        Text(String(format: "%.1fs total", metrics.totalTimeSeconds))
                    }
                }
                .font(.caption2)
                .foregroundStyle(.tertiary)
            }
        }
        .padding(.horizontal)
    }

    private var toolBubble: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: toolIcon)
                    .font(.caption)
                    .foregroundStyle(.orange)

                Text(toolCompletedLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Spacer()
            }

            if !message.text.isEmpty {
                Text(message.text)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(4)
                    .textSelection(.enabled)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 2)
    }

    private var visibleText: String {
        message.displayText ?? message.text
    }

    @ViewBuilder
    private var messageTextView: some View {
        if message.role == .assistant, let markdown = renderedMarkdown {
            Text(markdown)
                .font(.footnote)
                .foregroundStyle(.primary)
                .textSelection(.enabled)
        } else {
            Text(visibleText)
                .font(message.role == .assistant ? .footnote : .footnote)
                .foregroundStyle(message.role == .user ? .white : .primary)
                .textSelection(.enabled)
        }
    }

    private var renderedMarkdown: AttributedString? {
        guard !visibleText.isEmpty else { return nil }
        return try? AttributedString(
            markdown: visibleText,
            options: AttributedString.MarkdownParsingOptions(
                interpretedSyntax: .full
            )
        )
    }

    private var toolCompletedLabel: String {
        switch message.toolName {
        case "web_search": return "Searched the web"
        case "url_fetch": return "Fetched URL"
        default: return "Used \(message.toolName ?? "tool")"
        }
    }

    private var toolIcon: String {
        switch message.toolName {
        case "web_search": return "magnifyingglass"
        case "url_fetch": return "globe"
        default: return "wrench"
        }
    }
}
