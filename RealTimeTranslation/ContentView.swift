import SwiftUI
import AVFoundation

struct ContentView: View {
    @StateObject private var translationPipeline = TranslationPipeline()
    @State private var isTranslating = false
    @State private var sourceLanguage: Language = .spanish
    @State private var targetLanguage: Language = .english
    @State private var recognizedText = ""
    @State private var translatedText = ""
    @State private var showingLanguageSelector = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Header
                VStack(spacing: 8) {
                    Text("Real-Time Translator")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                        .foregroundColor(.primary)
                    
                    Text("Speak naturally, get instant translations")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding(.top)
                
                // Language Selection
                HStack(spacing: 20) {
                    LanguageButton(
                        language: sourceLanguage,
                        title: "From",
                        action: { showingLanguageSelector = true }
                    )
                    
                    Image(systemName: "arrow.right")
                        .font(.title2)
                        .foregroundColor(.blue)
                    
                    LanguageButton(
                        language: targetLanguage,
                        title: "To",
                        action: { showingLanguageSelector = true }
                    )
                }
                .padding(.horizontal)
                
                // Translation Status
                VStack(spacing: 16) {
                    // Recognized Text
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Recognized")
                                .font(.headline)
                                .foregroundColor(.primary)
                            Spacer()
                            Text(sourceLanguage.displayName)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Text(recognizedText.isEmpty ? "Listening..." : recognizedText)
                            .font(.body)
                            .foregroundColor(recognizedText.isEmpty ? .secondary : .primary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding()
                            .background(Color(.systemGray6))
                            .cornerRadius(12)
                    }
                    
                    // Translated Text
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("Translation")
                                .font(.headline)
                                .foregroundColor(.primary)
                            Spacer()
                            Text(targetLanguage.displayName)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        
                        Text(translatedText.isEmpty ? "Waiting for speech..." : translatedText)
                            .font(.body)
                            .foregroundColor(translatedText.isEmpty ? .secondary : .primary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding()
                            .background(Color(.systemGray6))
                            .cornerRadius(12)
                    }
                }
                .padding(.horizontal)
                
                Spacer()
                
                // Control Button
                Button(action: toggleTranslation) {
                    HStack(spacing: 12) {
                        Image(systemName: isTranslating ? "stop.circle.fill" : "mic.circle.fill")
                            .font(.title2)
                        
                        Text(isTranslating ? "Stop Translation" : "Start Translation")
                            .font(.headline)
                    }
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(isTranslating ? Color.red : Color.blue)
                    .cornerRadius(16)
                }
                .padding(.horizontal)
                .padding(.bottom)
            }
            .navigationBarHidden(true)
        }
        .onAppear {
            setupTranslationPipeline()
        }
        .sheet(isPresented: $showingLanguageSelector) {
            LanguageSelectorView(
                sourceLanguage: $sourceLanguage,
                targetLanguage: $targetLanguage
            )
        }
    }
    
    private func setupTranslationPipeline() {
        translationPipeline.onRecognizedText = { text in
            DispatchQueue.main.async {
                self.recognizedText = text
            }
        }
        
        translationPipeline.onTranslatedText = { text in
            DispatchQueue.main.async {
                self.translatedText = text
            }
        }
        
        translationPipeline.onError = { error in
            DispatchQueue.main.async {
                print("Translation error: \(error)")
            }
        }
    }
    
    private func toggleTranslation() {
        if isTranslating {
            translationPipeline.stopTranslation()
            isTranslating = false
        } else {
            translationPipeline.startTranslation(
                from: sourceLanguage,
                to: targetLanguage
            )
            isTranslating = true
        }
    }
}

struct LanguageButton: View {
    let language: Language
    let title: String
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 4) {
                Text(title)
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                Text(language.displayName)
                    .font(.headline)
                    .foregroundColor(.primary)
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color(.systemGray6))
            .cornerRadius(12)
        }
    }
}

struct LanguageSelectorView: View {
    @Binding var sourceLanguage: Language
    @Binding var targetLanguage: Language
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        NavigationView {
            List {
                Section("Source Language") {
                    ForEach(Language.allCases, id: \.self) { language in
                        Button(action: {
                            sourceLanguage = language
                        }) {
                            HStack {
                                Text(language.displayName)
                                    .foregroundColor(.primary)
                                Spacer()
                                if sourceLanguage == language {
                                    Image(systemName: "checkmark")
                                        .foregroundColor(.blue)
                                }
                            }
                        }
                    }
                }
                
                Section("Target Language") {
                    ForEach(Language.allCases, id: \.self) { language in
                        Button(action: {
                            targetLanguage = language
                        }) {
                            HStack {
                                Text(language.displayName)
                                    .foregroundColor(.primary)
                                Spacer()
                                if targetLanguage == language {
                                    Image(systemName: "checkmark")
                                        .foregroundColor(.blue)
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Select Languages")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        presentationMode.wrappedValue.dismiss()
                    }
                }
            }
        }
    }
}

#Preview {
    ContentView()
} 