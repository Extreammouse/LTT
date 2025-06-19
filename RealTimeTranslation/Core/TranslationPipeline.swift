import Foundation
import AVFoundation
import Speech

class TranslationPipeline: NSObject, ObservableObject {
    // MARK: - Public Properties
    
    var onRecognizedText: ((String) -> Void)?
    var onTranslatedText: ((String) -> Void)?
    var onError: ((Error) -> Void)?
    
    @Published var isTranslating = false
    @Published var currentSourceLanguage: Language = .spanish
    @Published var currentTargetLanguage: Language = .english
    
    // MARK: - Private Properties
    
    private let audioProcessor = AudioProcessor()
    private let speechRecognizer = SpeechRecognizer()
    private let translator = Translator()
    private let speechSynthesizer = SpeechSynthesizer()
    
    private var recognizedTextBuffer = ""
    private var lastTranslationTime = Date()
    private var translationDebounceTimer: Timer?
    
    private let processingQueue = DispatchQueue(label: "translation.pipeline", qos: .userInteractive)
    private let debounceInterval: TimeInterval = 1.0 // 1 second debounce
    
    // MARK: - Initialization
    
    override init() {
        super.init()
        setupComponents()
    }
    
    deinit {
        stopTranslation()
    }
    
    // MARK: - Public Methods
    
    func startTranslation(from sourceLanguage: Language, to targetLanguage: Language) {
        guard !isTranslating else { return }
        
        currentSourceLanguage = sourceLanguage
        currentTargetLanguage = targetLanguage
        
        do {
            // Request permissions
            try requestPermissions()
            
            // Load models
            loadModels()
            
            // Start audio processing
            try audioProcessor.startRecording()
            
            // Start speech recognition
            try speechRecognizer.startRecognition(language: sourceLanguage)
            
            isTranslating = true
            
        } catch {
            onError?(error)
        }
    }
    
    func stopTranslation() {
        guard isTranslating else { return }
        
        // Stop all components
        audioProcessor.stopRecording()
        speechRecognizer.stopRecognition()
        speechSynthesizer.stopSpeaking()
        
        // Cancel any pending timers
        translationDebounceTimer?.invalidate()
        translationDebounceTimer = nil
        
        // Clear buffers
        recognizedTextBuffer = ""
        
        isTranslating = false
    }
    
    func updateLanguages(source: Language, target: Language) {
        currentSourceLanguage = source
        currentTargetLanguage = target
        
        // Reload models for new language pair
        loadModels()
        
        // Update speech recognizer language if currently translating
        if isTranslating {
            do {
                speechRecognizer.stopRecognition()
                try speechRecognizer.startRecognition(language: source)
            } catch {
                onError?(error)
            }
        }
    }
    
    // MARK: - Private Methods
    
    private func setupComponents() {
        // Set up audio processor
        audioProcessor.delegate = self
        
        // Set up speech recognizer
        speechRecognizer.delegate = self
        
        // Set up translator
        translator.delegate = self
        
        // Set up speech synthesizer
        speechSynthesizer.delegate = self
    }
    
    private func requestPermissions() throws {
        // Request microphone permission
        let audioSession = AVAudioSession.sharedInstance()
        switch audioSession.recordPermission {
        case .granted:
            break
        case .denied:
            throw TranslationPipelineError.microphonePermissionDenied
        case .undetermined:
            audioSession.requestRecordPermission { [weak self] granted in
                if !granted {
                    DispatchQueue.main.async {
                        self?.onError?(TranslationPipelineError.microphonePermissionDenied)
                    }
                }
            }
        @unknown default:
            throw TranslationPipelineError.microphonePermissionDenied
        }
        
        // Request speech recognition permission
        SFSpeechRecognizer.requestAuthorization { [weak self] status in
            DispatchQueue.main.async {
                switch status {
                case .authorized:
                    break
                case .denied, .restricted, .notDetermined:
                    self?.onError?(TranslationPipelineError.speechRecognitionPermissionDenied)
                @unknown default:
                    self?.onError?(TranslationPipelineError.speechRecognitionPermissionDenied)
                }
            }
        }
    }
    
    private func loadModels() {
        // Load translation model
        translator.loadModel(for: currentSourceLanguage, target: currentTargetLanguage)
        
        // Load speech recognition model
        speechRecognizer.loadCustomModel(for: currentSourceLanguage)
    }
    
    private func processRecognizedText(_ text: String, isFinal: Bool) {
        // Update recognized text buffer
        if isFinal {
            recognizedTextBuffer = text
        } else {
            // For partial results, append to buffer
            if !text.isEmpty && !recognizedTextBuffer.contains(text) {
                recognizedTextBuffer = text
            }
        }
        
        // Notify UI of recognized text
        onRecognizedText?(recognizedTextBuffer)
        
        // Debounce translation to avoid too frequent API calls
        debounceTranslation()
    }
    
    private func debounceTranslation() {
        // Cancel existing timer
        translationDebounceTimer?.invalidate()
        
        // Create new timer
        translationDebounceTimer = Timer.scheduledTimer(withTimeInterval: debounceInterval, repeats: false) { [weak self] _ in
            self?.performTranslation()
        }
    }
    
    private func performTranslation() {
        guard !recognizedTextBuffer.isEmpty else { return }
        
        // Check if enough time has passed since last translation
        let timeSinceLastTranslation = Date().timeIntervalSince(lastTranslationTime)
        guard timeSinceLastTranslation >= debounceInterval else { return }
        
        // Perform translation
        translator.translate(
            text: recognizedTextBuffer,
            from: currentSourceLanguage,
            to: currentTargetLanguage
        )
        
        lastTranslationTime = Date()
    }
    
    private func processTranslatedText(_ text: String) {
        // Notify UI of translated text
        onTranslatedText?(text)
        
        // Optionally speak the translated text
        speakTranslatedText(text)
    }
    
    private func speakTranslatedText(_ text: String) {
        // Only speak if not already speaking
        guard !speechSynthesizer.isSpeaking else { return }
        
        speechSynthesizer.speak(text: text, language: currentTargetLanguage)
    }
    
    // MARK: - Error Handling
    
    private func handleError(_ error: Error) {
        DispatchQueue.main.async { [weak self] in
            self?.onError?(error)
        }
    }
}

// MARK: - AudioProcessorDelegate

extension TranslationPipeline: AudioProcessorDelegate {
    func audioProcessor(_ processor: AudioProcessor, didReceiveAudioBuffer buffer: [Float]) {
        // Pass audio buffer to speech recognizer for custom model processing
        speechRecognizer.processAudioBuffer(buffer)
    }
    
    func audioProcessor(_ processor: AudioProcessor, didEncounterError error: Error) {
        handleError(error)
    }
}

// MARK: - SpeechRecognizerDelegate

extension TranslationPipeline: SpeechRecognizerDelegate {
    func speechRecognizer(_ recognizer: SpeechRecognizer, didRecognizeText text: String, isFinal: Bool) {
        processRecognizedText(text, isFinal: isFinal)
    }
    
    func speechRecognizer(_ recognizer: SpeechRecognizer, didEncounterError error: Error) {
        handleError(error)
    }
}

// MARK: - TranslatorDelegate

extension TranslationPipeline: TranslatorDelegate {
    func translator(_ translator: Translator, didTranslateText text: String, from sourceLanguage: Language, to targetLanguage: Language) {
        processTranslatedText(text)
    }
    
    func translator(_ translator: Translator, didEncounterError error: Error) {
        handleError(error)
    }
}

// MARK: - SpeechSynthesizerDelegate

extension TranslationPipeline: SpeechSynthesizerDelegate {
    func speechSynthesizer(_ synthesizer: SpeechSynthesizer, didStartSpeaking text: String) {
        // Speech synthesis started
    }
    
    func speechSynthesizer(_ synthesizer: SpeechSynthesizer, didFinishSpeaking text: String) {
        // Speech synthesis finished
    }
    
    func speechSynthesizer(_ synthesizer: SpeechSynthesizer, didEncounterError error: Error) {
        handleError(error)
    }
}

// MARK: - Error Types

enum TranslationPipelineError: Error, LocalizedError {
    case microphonePermissionDenied
    case speechRecognitionPermissionDenied
    case audioProcessingFailed
    case speechRecognitionFailed
    case translationFailed
    case speechSynthesisFailed
    case modelLoadingFailed
    
    var errorDescription: String? {
        switch self {
        case .microphonePermissionDenied:
            return "Microphone permission is required for real-time translation"
        case .speechRecognitionPermissionDenied:
            return "Speech recognition permission is required for real-time translation"
        case .audioProcessingFailed:
            return "Failed to process audio input"
        case .speechRecognitionFailed:
            return "Failed to recognize speech"
        case .translationFailed:
            return "Failed to translate text"
        case .speechSynthesisFailed:
            return "Failed to synthesize speech"
        case .modelLoadingFailed:
            return "Failed to load required models"
        }
    }
}

// MARK: - Performance Monitoring

extension TranslationPipeline {
    func getPerformanceMetrics() -> PerformanceMetrics {
        return PerformanceMetrics(
            audioLatency: calculateAudioLatency(),
            recognitionLatency: calculateRecognitionLatency(),
            translationLatency: calculateTranslationLatency(),
            synthesisLatency: calculateSynthesisLatency()
        )
    }
    
    private func calculateAudioLatency() -> TimeInterval {
        // Calculate audio processing latency
        return 0.1 // Placeholder
    }
    
    private func calculateRecognitionLatency() -> TimeInterval {
        // Calculate speech recognition latency
        return 0.5 // Placeholder
    }
    
    private func calculateTranslationLatency() -> TimeInterval {
        // Calculate translation latency
        return 0.3 // Placeholder
    }
    
    private func calculateSynthesisLatency() -> TimeInterval {
        // Calculate speech synthesis latency
        return 0.2 // Placeholder
    }
}

struct PerformanceMetrics {
    let audioLatency: TimeInterval
    let recognitionLatency: TimeInterval
    let translationLatency: TimeInterval
    let synthesisLatency: TimeInterval
    
    var totalLatency: TimeInterval {
        return audioLatency + recognitionLatency + translationLatency + synthesisLatency
    }
} 