import Foundation

enum BraveSearchService {

    static func search(query: String, apiKey: String, count: Int = 5) async -> String {
        guard !query.isEmpty else { return "Error: Empty search query" }

        guard var components = URLComponents(string: "https://api.search.brave.com/res/v1/web/search") else {
            return "Error: Invalid URL"
        }
        components.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "count", value: String(count)),
        ]

        guard let url = components.url else {
            return "Error: Could not build search URL"
        }

        var request = URLRequest(url: url)
        request.setValue(apiKey, forHTTPHeaderField: "X-Subscription-Token")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.timeoutInterval = 15

        do {
            let (data, response) = try await URLSession.shared.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                return "Error: Invalid response"
            }
            guard httpResponse.statusCode == 200 else {
                return "Error: Search API returned status \(httpResponse.statusCode)"
            }

            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            guard let webResults = json?["web"] as? [String: Any],
                  let results = webResults["results"] as? [[String: Any]] else {
                return "No results found for '\(query)'"
            }

            var output = "Search results for '\(query)':\n\n"
            for (i, result) in results.prefix(count).enumerated() {
                let title = result["title"] as? String ?? "No title"
                let description = result["description"] as? String ?? ""
                let resultUrl = result["url"] as? String ?? ""
                output += "\(i + 1). \(title)\n"
                if !description.isEmpty { output += "   \(description)\n" }
                if !resultUrl.isEmpty { output += "   URL: \(resultUrl)\n" }
                output += "\n"
            }
            return output.trimmingCharacters(in: .whitespacesAndNewlines)
        } catch {
            return "Error: \(error.localizedDescription)"
        }
    }
}
