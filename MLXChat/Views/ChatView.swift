import SwiftUI
import PhotosUI

struct ChatView: View {
    @State private var viewModel = ChatViewModel()
    @State private var inputText = ""
    @State private var pendingImage: UIImage?
    @State private var selectedPhoto: PhotosPickerItem?
    @State private var showCamera = false
    @State private var showPhotoPicker = false
    @State private var showModelPicker = false
    @State private var settings = SettingsManager.shared
    @FocusState private var inputFocused: Bool

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Model header + thinking toggle
                HStack {
                    Button {
                        showModelPicker = true
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: settings.loadedModelName != nil ? "circle.fill" : "circle")
                                .font(.caption2)
                                .foregroundStyle(settings.loadedModelName != nil ? .green : .secondary)
                            if let name = settings.loadedModelName {
                                Text(name)
                                    .lineLimit(1)
                                    .font(.subheadline)
                            } else {
                                Text("Select a model")
                                    .font(.subheadline)
                                    .foregroundStyle(.secondary)
                            }
                            Image(systemName: "chevron.down")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .buttonStyle(.plain)

                    Spacer()

                    if viewModel.isThinkingModel {
                        Toggle(isOn: $viewModel.thinkingEnabled) {
                            Label("Think", systemImage: "brain")
                                .font(.subheadline)
                        }
                        .toggleStyle(.button)
                        .buttonStyle(.bordered)
                        .tint(viewModel.thinkingEnabled ? .purple : .gray)
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 8)

                Divider()

                // Messages
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 8) {
                            ForEach(viewModel.messages) { message in
                                ChatBubbleView(message: message)
                                    .id(message.id)
                            }

                            if viewModel.isGenerating {
                                HStack {
                                    ProgressView()
                                    if let status = viewModel.statusMessage {
                                        Text(status)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                    Spacer()
                                }
                                .padding(.horizontal)
                                .id("generating")
                            }
                        }
                        .padding(.vertical, 8)
                    }
                    .scrollDismissesKeyboard(.interactively)
                    .onTapGesture {
                        inputFocused = false
                    }
                    .onChange(of: viewModel.messages.count) {
                        withAnimation {
                            if let last = viewModel.messages.last {
                                proxy.scrollTo(last.id, anchor: .bottom)
                            }
                        }
                    }
                    .onChange(of: viewModel.messages.last?.text) {
                        if let last = viewModel.messages.last {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }

                if let error = viewModel.error {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .padding(.horizontal)
                        .padding(.vertical, 4)
                }

            }
            .safeAreaInset(edge: .bottom) {
                // Input bar in safe area inset so it stays above keyboard
                VStack(spacing: 0) {
                    Divider()

                    // Pending image preview
                    if let image = pendingImage {
                        HStack {
                            Image(uiImage: image)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(height: 60)
                                .clipShape(RoundedRectangle(cornerRadius: 6))

                            Button {
                                pendingImage = nil
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundStyle(.secondary)
                            }

                            Spacer()
                        }
                        .padding(.horizontal)
                        .padding(.top, 6)
                    }

                    HStack(spacing: 8) {
                        Menu {
                            Button {
                                showPhotoPicker = true
                            } label: {
                                Label("Photo Library", systemImage: "photo.on.rectangle")
                            }

                            if UIImagePickerController.isSourceTypeAvailable(.camera) {
                                Button {
                                    showCamera = true
                                } label: {
                                    Label("Take Photo", systemImage: "camera")
                                }
                            }
                        } label: {
                            Image(systemName: "plus.circle.fill")
                                .font(.title2)
                                .foregroundStyle(.blue)
                        }

                        TextField("Message...", text: $inputText, axis: .vertical)
                            .textFieldStyle(.roundedBorder)
                            .lineLimit(1...5)
                            .focused($inputFocused)
                            .submitLabel(.send)
                            .onSubmit {
                                if canSend { sendCurrentMessage() }
                            }

                        if viewModel.isGenerating {
                            Button {
                                viewModel.stopGeneration()
                            } label: {
                                Image(systemName: "stop.circle.fill")
                                    .font(.title2)
                                    .foregroundStyle(.red)
                            }
                        } else {
                            Button {
                                sendCurrentMessage()
                            } label: {
                                Image(systemName: "arrow.up.circle.fill")
                                    .font(.title2)
                                    .foregroundStyle(canSend ? .blue : .gray)
                            }
                            .disabled(!canSend)
                        }
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 8)
                }
                .background(.ultraThinMaterial)
            }
            .navigationTitle("Chat")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Menu {
                        Button(role: .destructive) {
                            viewModel.clearSession()
                        } label: {
                            Label("Clear Chat", systemImage: "trash")
                        }

                    } label: {
                        Image(systemName: "ellipsis.circle")
                    }
                }

            }
            .photosPicker(isPresented: $showPhotoPicker, selection: $selectedPhoto, matching: .images)
            .onChange(of: selectedPhoto) {
                loadPhoto()
            }
            .fullScreenCover(isPresented: $showCamera) {
                CameraView(image: $pendingImage)
                    .ignoresSafeArea()
            }
            .sheet(isPresented: $showModelPicker) {
                ModelPickerView(viewModel: viewModel)
            }
        }
    }

    private var canSend: Bool {
        !viewModel.isGenerating && settings.loadedModelId != nil &&
        (!inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || pendingImage != nil)
    }

    private func sendCurrentMessage() {
        inputFocused = false
        let text = inputText
        let image = pendingImage
        inputText = ""
        pendingImage = nil
        selectedPhoto = nil

        viewModel.generationTask = Task {
            await viewModel.sendMessage(text: text, image: image)
        }
    }

    private func loadPhoto() {
        guard let item = selectedPhoto else { return }
        Task {
            if let data = try? await item.loadTransferable(type: Data.self),
               let image = UIImage(data: data) {
                pendingImage = image
            }
        }
    }
}

// MARK: - Camera

struct CameraView: UIViewControllerRepresentable {
    @Binding var image: UIImage?
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: CameraView

        init(_ parent: CameraView) {
            self.parent = parent
        }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            parent.image = info[.originalImage] as? UIImage
            parent.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.dismiss()
        }
    }
}
