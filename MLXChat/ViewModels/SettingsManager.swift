import Foundation
import CoreLocation
import Observation
import UIKit

struct CachedModelInfo: Identifiable {
    let id: String  // hfId e.g. "mlx-community/Qwen3.5-2B-MLX-8bit"
    let displayName: String
    let sizeBytes: Int64
    let path: URL

    var sizeFormatted: String {
        let gb = Double(sizeBytes) / (1024 * 1024 * 1024)
        if gb >= 1.0 {
            return String(format: "%.1f GB", gb)
        } else {
            let mb = Double(sizeBytes) / (1024 * 1024)
            return String(format: "%.0f MB", mb)
        }
    }
}

@MainActor
@Observable
final class SettingsManager {
    static let shared = SettingsManager()

    @ObservationIgnored
    private var modelDownloadTasks: [String: Task<Void, Never>] = [:]

    var loadedModelId: String? {
        didSet { UserDefaults.standard.set(loadedModelId, forKey: "loadedModelId") }
    }

    var loadedModelName: String? {
        didSet { UserDefaults.standard.set(loadedModelName, forKey: "loadedModelName") }
    }

    var cachedModels: [CachedModelInfo] = []
    var modelDownloadProgress: [String: Double] = [:]

    var sharedEngine: MLXEngine?
    private let locationProvider = PromptLocationProvider()

    var contextSize: Int = 4096 {
        didSet { UserDefaults.standard.set(contextSize, forKey: "contextSize") }
    }

    var maxImageDimension: Int = 256 {
        didSet { UserDefaults.standard.set(maxImageDimension, forKey: "maxImageDimension") }
    }

    var jpegQuality: Double = 0.7 {
        didSet { UserDefaults.standard.set(jpegQuality, forKey: "jpegQuality") }
    }

    /// GPU memory limit in GB — applied on next engine creation (requires app restart)
    var gpuMemoryLimitGB: Int = 5 {
        didSet { UserDefaults.standard.set(gpuMemoryLimitGB, forKey: "gpuMemoryLimitGB") }
    }

    /// Maximum model size shown for new downloads. `0` means no filter.
    var maxModelDownloadSizeGB: Int = 0 {
        didSet { UserDefaults.standard.set(maxModelDownloadSizeGB, forKey: "maxModelDownloadSizeGB") }
    }

    var toolsEnabled: Bool = true {
        didSet { UserDefaults.standard.set(toolsEnabled, forKey: "toolsEnabled") }
    }

    var braveAPIKey: String = "" {
        didSet { UserDefaults.standard.set(braveAPIKey, forKey: "braveAPIKey") }
    }

    var streamingEnabled: Bool = true {
        didSet { UserDefaults.standard.set(streamingEnabled, forKey: "streamingEnabled") }
    }

    static let defaultSystemPrompt = "You are a helpful assistant. Today is {today} and the time is {time} in the {timezone} timezone. Your rough location is {location} and your coordinates are {coordinates}. Use British English spelling. Do not use this information unless relevant to the user's question."

    var systemPrompt: String = SettingsManager.defaultSystemPrompt {
        didSet { UserDefaults.standard.set(systemPrompt, forKey: "systemPrompt") }
    }

    /// Substitutes {variables} in the system prompt with live values
    func processedSystemPrompt() -> String {
        locationProvider.requestLocationIfNeeded()
        return substituteVariables(in: systemPrompt)
    }

    private func substituteVariables(in prompt: String) -> String {
        var result = prompt

        let variables: [String: () -> String] = [
            "today": {
                let f = DateFormatter()
                f.dateFormat = "EEEE, MMMM d, yyyy"
                f.locale = Locale(identifier: "en_GB")
                return f.string(from: Date())
            },
            "date": {
                let f = DateFormatter()
                f.dateFormat = "dd MMM yyyy"
                f.locale = Locale(identifier: "en_GB")
                return f.string(from: Date())
            },
            "time": {
                let f = DateFormatter()
                f.dateFormat = "HH:mm:ss"
                return f.string(from: Date())
            },
            "datetime": {
                let f = DateFormatter()
                f.dateFormat = "dd MMM yyyy HH:mm:ss"
                f.locale = Locale(identifier: "en_GB")
                return f.string(from: Date())
            },
            "timestamp": {
                let f = DateFormatter()
                f.dateFormat = "yyyy-MM-dd HH:mm:ss"
                return f.string(from: Date())
            },
            "unixtime": { String(Int(Date().timeIntervalSince1970)) },
            "location": { self.locationProvider.locationName },
            "address": { self.locationProvider.locationName },
            "coordinates": { self.locationProvider.coordinateString },
            "latitude": { self.locationProvider.latitudeString },
            "longitude": { self.locationProvider.longitudeString },
            "timezone": { TimeZone.current.identifier },
            "locale": { Locale.current.identifier },
            "device": { UIDevice.current.model },
            "system": { UIDevice.current.systemName },
            "version": { UIDevice.current.systemVersion },
            "username": { NSUserName() },
        ]

        for (variable, getValue) in variables {
            let pattern = "{\(variable)}"
            if result.contains(pattern) {
                result = result.replacingOccurrences(of: pattern, with: getValue())
            }
        }
        return result
    }

    var availableMemoryGB: Double {
        Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024) * 0.6
    }

    var totalMemoryGB: Double {
        Double(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024 * 1024)
    }

    var gpuMemoryLimitOptions: [Int] {
        let physicalGB = max(4, Int(floor(totalMemoryGB)))
        let recommendedMax = max(physicalGB - 2, 4)
        let cappedMax = min(recommendedMax, 24)
        let values = Array(3...cappedMax)
        return values.isEmpty ? [3, 4, 5, 6, 7, 8] : values
    }

    var modelDownloadLimitOptions: [Int] {
        [0, 1, 2, 3, 4, 6, 8, 12, 16, 24]
    }

    private(set) var downloadedModelIds: Set<String> = []

    private init() {
        let savedContext = UserDefaults.standard.integer(forKey: "contextSize")
        if [4096, 8192, 12288, 16384, 20480, 24576, 28672, 32768].contains(savedContext) {
            contextSize = savedContext
        } else {
            UserDefaults.standard.set(4096, forKey: "contextSize")
        }

        let savedDim = UserDefaults.standard.integer(forKey: "maxImageDimension")
        if savedDim >= 128 && savedDim <= 1024 {
            maxImageDimension = savedDim
        } else {
            UserDefaults.standard.set(256, forKey: "maxImageDimension")
        }

        let savedQuality = UserDefaults.standard.double(forKey: "jpegQuality")
        if savedQuality >= 0.1 && savedQuality <= 1.0 {
            jpegQuality = savedQuality
        } else {
            UserDefaults.standard.set(0.7, forKey: "jpegQuality")
        }

        let savedGPULimit = UserDefaults.standard.integer(forKey: "gpuMemoryLimitGB")
        if gpuMemoryLimitOptions.contains(savedGPULimit) {
            gpuMemoryLimitGB = savedGPULimit
        } else {
            UserDefaults.standard.set(5, forKey: "gpuMemoryLimitGB")
        }

        let savedDownloadLimit = UserDefaults.standard.integer(forKey: "maxModelDownloadSizeGB")
        if modelDownloadLimitOptions.contains(savedDownloadLimit) {
            maxModelDownloadSizeGB = savedDownloadLimit
        } else {
            UserDefaults.standard.set(0, forKey: "maxModelDownloadSizeGB")
        }

        if UserDefaults.standard.object(forKey: "toolsEnabled") != nil {
            toolsEnabled = UserDefaults.standard.bool(forKey: "toolsEnabled")
        }

        braveAPIKey = UserDefaults.standard.string(forKey: "braveAPIKey") ?? ""

        if UserDefaults.standard.object(forKey: "streamingEnabled") != nil {
            streamingEnabled = UserDefaults.standard.bool(forKey: "streamingEnabled")
        }

        systemPrompt = UserDefaults.standard.string(forKey: "systemPrompt") ?? Self.defaultSystemPrompt

        refreshCachedModels()

        let savedModelId = UserDefaults.standard.string(forKey: "loadedModelId")
        let savedModelName = UserDefaults.standard.string(forKey: "loadedModelName")

        if let id = savedModelId, !id.isEmpty {
            loadedModelId = id
            loadedModelName = savedModelName
        } else if let first = cachedModels.first {
            loadedModelId = first.id
            loadedModelName = first.displayName
            UserDefaults.standard.set(first.id, forKey: "loadedModelId")
            UserDefaults.standard.set(first.displayName, forKey: "loadedModelName")
        }
    }

    func isModelDownloaded(hfId: String) -> Bool {
        downloadedModelIds.contains(hfId)
    }

    func setModelDownloadProgress(hfId: String, progress: Double) {
        let clamped = min(max(progress, 0), 1)
        if clamped <= 0 {
            modelDownloadProgress.removeValue(forKey: hfId)
        } else if clamped >= 1 {
            modelDownloadProgress.removeValue(forKey: hfId)
        } else {
            modelDownloadProgress[hfId] = clamped
        }
    }

    var activeModelDownloads: [(hfId: String, progress: Double)] {
        modelDownloadProgress
            .map { (hfId: $0.key, progress: $0.value) }
            .sorted { lhs, rhs in
                let left = ModelRegistry.find(hfId: lhs.hfId)?.displayName ?? lhs.hfId
                let right = ModelRegistry.find(hfId: rhs.hfId)?.displayName ?? rhs.hfId
                return left.localizedCaseInsensitiveCompare(right) == .orderedAscending
            }
    }

    func registerModelDownloadTask(hfId: String, task: Task<Void, Never>) {
        modelDownloadTasks[hfId] = task
    }

    func finishModelDownloadTask(hfId: String) {
        modelDownloadTasks.removeValue(forKey: hfId)
    }

    func cancelModelDownload(hfId: String) {
        modelDownloadTasks[hfId]?.cancel()
        modelDownloadTasks.removeValue(forKey: hfId)
        setModelDownloadProgress(hfId: hfId, progress: 0)
    }

    func modelDisplayName(hfId: String) -> String {
        ModelRegistry.find(hfId: hfId)?.displayName ?? hfId
    }

    private var hubModelsDir: URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return documents.appending(path: "huggingface/models")
    }

    var totalCacheSize: String {
        let total = cachedModels.reduce(Int64(0)) { $0 + $1.sizeBytes }
        let gb = Double(total) / (1024 * 1024 * 1024)
        return String(format: "%.1f GB", gb)
    }

    var appVersionString: String {
        let bundle = Bundle.main
        let shortVersion = bundle.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?"
        let buildNumber = bundle.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "?"
        return "\(shortVersion) (\(buildNumber))"
    }

    var runtimeBuildMarker: String {
        "qwen3.5-fixes-2026-03-06-001"
    }

    func refreshCachedModels() {
        let fm = FileManager.default
        let modelsDir = hubModelsDir
        var models: [CachedModelInfo] = []

        guard let orgDirs = try? fm.contentsOfDirectory(at: modelsDir, includingPropertiesForKeys: nil) else {
            cachedModels = []
            downloadedModelIds = []
            return
        }

        for orgDir in orgDirs {
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: orgDir.path, isDirectory: &isDir), isDir.boolValue else { continue }
            let orgName = orgDir.lastPathComponent
            guard !orgName.hasPrefix(".") else { continue }

            guard let repoDirs = try? fm.contentsOfDirectory(at: orgDir, includingPropertiesForKeys: nil) else { continue }
            for repoDir in repoDirs {
                guard fm.fileExists(atPath: repoDir.path, isDirectory: &isDir), isDir.boolValue else { continue }
                let repoName = repoDir.lastPathComponent
                guard !repoName.hasPrefix(".") else { continue }

                let hfId = "\(orgName)/\(repoName)"
                let size = directorySize(at: repoDir)

                models.append(CachedModelInfo(
                    id: hfId,
                    displayName: repoName,
                    sizeBytes: size,
                    path: repoDir
                ))
            }
        }

        cachedModels = models.sorted { $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending }
        downloadedModelIds = Set(models.map(\.id))
    }

    func deleteModel(model: CachedModelInfo) {
        try? FileManager.default.removeItem(at: model.path)
        if loadedModelId == model.id {
            loadedModelId = nil
            loadedModelName = nil
        }
        refreshCachedModels()
    }

    private func directorySize(at url: URL) -> Int64 {
        let fm = FileManager.default
        var total: Int64 = 0
        guard let enumerator = fm.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]) else { return 0 }
        for case let fileURL as URL in enumerator {
            if let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize {
                total += Int64(size)
            }
        }
        return total
    }
}

@MainActor
private final class PromptLocationProvider: NSObject, @preconcurrency CLLocationManagerDelegate {
    private let manager = CLLocationManager()
    private let geocoder = CLGeocoder()
    private(set) var currentLocation: CLLocation?
    private(set) var locationName = "Unknown"
    private var hasRequested = false

    override init() {
        super.init()
        manager.delegate = self
        manager.desiredAccuracy = kCLLocationAccuracyKilometer
    }

    func requestLocationIfNeeded() {
        guard !hasRequested else { return }
        hasRequested = true

        switch manager.authorizationStatus {
        case .notDetermined:
            manager.requestWhenInUseAuthorization()
        case .authorizedWhenInUse, .authorizedAlways:
            manager.requestLocation()
        case .restricted, .denied:
            locationName = "Unknown"
        @unknown default:
            locationName = "Unknown"
        }
    }

    var coordinateString: String {
        guard let currentLocation else { return "Unknown" }
        return String(format: "%.4f, %.4f", currentLocation.coordinate.latitude, currentLocation.coordinate.longitude)
    }

    var latitudeString: String {
        guard let currentLocation else { return "Unknown" }
        return String(currentLocation.coordinate.latitude)
    }

    var longitudeString: String {
        guard let currentLocation else { return "Unknown" }
        return String(currentLocation.coordinate.longitude)
    }

    func locationManagerDidChangeAuthorization(_ manager: CLLocationManager) {
        switch manager.authorizationStatus {
        case .authorizedWhenInUse, .authorizedAlways:
            manager.requestLocation()
        case .restricted, .denied:
            locationName = "Unknown"
        case .notDetermined:
            break
        @unknown default:
            locationName = "Unknown"
        }
    }

    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        guard let location = locations.last else { return }
        currentLocation = location
        geocoder.reverseGeocodeLocation(location) { [weak self] placemarks, _ in
            Task { @MainActor in
                guard let self else { return }
                guard let placemark = placemarks?.first else {
                    self.locationName = self.coordinateString
                    return
                }

                var components: [String] = []
                if let locality = placemark.locality {
                    components.append(locality)
                }
                if let administrativeArea = placemark.administrativeArea {
                    components.append(administrativeArea)
                }
                if let country = placemark.country {
                    components.append(country)
                }

                let resolved = components.joined(separator: ", ")
                self.locationName = resolved.isEmpty ? self.coordinateString : resolved
            }
        }
    }

    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        if currentLocation == nil {
            locationName = "Unknown"
        }
    }
}
