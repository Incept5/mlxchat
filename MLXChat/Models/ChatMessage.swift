import Foundation
import UIKit

enum ChatRole: Sendable {
    case user
    case assistant
    case system
    case tool
}

struct ChatMessage: Identifiable, Sendable {
    let id = UUID()
    let role: ChatRole
    var text: String
    var displayText: String?
    var thinkingText: String?
    var isThinkingInProgress: Bool = false
    var image: UIImage?
    var metrics: GenerationMetrics?
    let toolName: String?
    let timestamp = Date()

    init(
        role: ChatRole,
        text: String,
        displayText: String? = nil,
        thinkingText: String? = nil,
        isThinkingInProgress: Bool = false,
        image: UIImage? = nil,
        metrics: GenerationMetrics? = nil,
        toolName: String? = nil
    ) {
        self.role = role
        self.text = text
        self.displayText = displayText
        self.thinkingText = thinkingText
        self.isThinkingInProgress = isThinkingInProgress
        self.image = image
        self.metrics = metrics
        self.toolName = toolName
    }
}
