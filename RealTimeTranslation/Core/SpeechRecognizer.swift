import Foundation
import Speech
import CoreML
import Accelerate

protocol SpeechRecognizerDelegate: AnyObject {
    func speechRecognizer(_ recognizer: SpeechRecognizer, didRecognizeText text: String, isFinal: Bool)
    func speechRecognizer(_ recognizer: SpeechRecognizer, didEncounterError error: Error)
}

class SpeechRecognizer: NSObject {
    weak var delegate: SpeechRecognizerDelegate?
    
    private var speechRecognizer: SFSpeechRecognizer?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var audioEngine: AVAudioEngine?
    
    private var currentLanguage: Language = .english
    private var isRecognizing = false
    private var customModel: MLModel?
    
    private let processingQueue = DispatchQueue(label: "speech.recognition", qos: .userInteractive)
    
    override init() {
        super.init()
        setupSpeechRecognizer()
    }
    
    deinit {
        stopRecognition()
    }
    
    // MARK: - Public Methods
    
    func startRecognition(language: Language) throws {
        guard !isRecognizing else { return }
        
        currentLanguage = language
        
        do {
            try setupAudioEngine()
            try setupRecognitionRequest()
            try startAudioEngine()
            isRecognizing = true
        } catch {
            delegate?.speechRecognizer(self, didEncounterError: error)
            throw error
        }
    }
    
    func stopRecognition() {
        guard isRecognizing else { return }
        
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        
        isRecognizing = false
    }
    
    func processAudioBuffer(_ audioBuffer: [Float]) {
        guard isRecognizing else { return }
        
        processingQueue.async { [weak self] in
            self?.performCustomRecognition(on: audioBuffer)
        }
    }
    
    // MARK: - Private Methods
    
    private func setupSpeechRecognizer() {
        speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: currentLanguage.speechRecognitionLanguage))
        speechRecognizer?.delegate = self
    }
    
    private func setupAudioEngine() throws {
        audioEngine = AVAudioEngine()
        guard let audioEngine = audioEngine else {
            throw SpeechRecognizerError.audioEngineCreationFailed
        }
        
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
        }
    }
    
    private func setupRecognitionRequest() throws {
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else {
            throw SpeechRecognizerError.recognitionRequestCreationFailed
        }
        
        recognitionRequest.shouldReportPartialResults = true
        
        // Update speech recognizer for current language
        speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: currentLanguage.speechRecognitionLanguage))
        speechRecognizer?.delegate = self
        
        guard let speechRecognizer = speechRecognizer, speechRecognizer.isAvailable else {
            throw SpeechRecognizerError.speechRecognizerNotAvailable
        }
        
        recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            guard let self = self else { return }
            
            if let error = error {
                self.delegate?.speechRecognizer(self, didEncounterError: error)
                return
            }
            
            if let result = result {
                let text = result.bestTranscription.formattedString
                let isFinal = result.isFinal
                self.delegate?.speechRecognizer(self, didRecognizeText: text, isFinal: isFinal)
            }
        }
    }
    
    private func startAudioEngine() throws {
        guard let audioEngine = audioEngine else {
            throw SpeechRecognizerError.audioEngineNotInitialized
        }
        
        audioEngine.prepare()
        try audioEngine.start()
    }
    
    // MARK: - Custom Model Recognition
    
    private func performCustomRecognition(on audioBuffer: [Float]) {
        // This is where you'd use your custom trained model
        // For now, we'll use a simplified approach
        
        let features = extractFeatures(from: audioBuffer)
        let recognizedText = recognizeTextFromFeatures(features)
        
        if !recognizedText.isEmpty {
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.delegate?.speechRecognizer(self, didRecognizeText: recognizedText, isFinal: false)
            }
        }
    }
    
    private func extractFeatures(from audioBuffer: [Float]) -> [Float] {
        // Extract MFCC features or other audio features
        // This is a simplified implementation
        
        let frameSize = 512
        let hopSize = 256
        var features: [Float] = []
        
        for i in stride(from: 0, to: audioBuffer.count - frameSize, by: hopSize) {
            let frame = Array(audioBuffer[i..<min(i + frameSize, audioBuffer.count)])
            let frameFeatures = extractFrameFeatures(from: frame)
            features.append(contentsOf: frameFeatures)
        }
        
        return features
    }
    
    private func extractFrameFeatures(from frame: [Float]) -> [Float] {
        // Simplified feature extraction
        // In practice, you'd use FFT, mel-filterbank, etc.
        
        let numFeatures = 13
        var features = [Float](repeating: 0.0, count: numFeatures)
        
        // Energy-based features
        let energy = frame.map { $0 * $0 }.reduce(0, +)
        let logEnergy = log(max(energy, 1e-10))
        
        // Spectral features (simplified)
        for i in 0..<numFeatures {
            let frequencyBin = Float(i) / Float(numFeatures)
            let spectralEnergy = frame.enumerated().map { index, sample in
                sample * cos(2.0 * Float.pi * frequencyBin * Float(index))
            }.reduce(0, +)
            
            features[i] = log(max(abs(spectralEnergy), 1e-10))
        }
        
        return features
    }
    
    private func recognizeTextFromFeatures(_ features: [Float]) -> String {
        // This is where you'd use your trained model
        // For now, we'll return a placeholder
        
        // In a real implementation, you'd:
        // 1. Load your trained Core ML model
        // 2. Preprocess features to match model input
        // 3. Run inference
        // 4. Post-process output to get text
        
        return ""
    }
    
    // MARK: - Model Loading
    
    func loadCustomModel(for language: Language) {
        let model = SpeechRecognitionModel(language: language)
        
        guard let modelURL = model.modelURL else {
            print("Custom model not available for \(language.displayName)")
            return
        }
        
        do {
            customModel = try MLModel(contentsOf: modelURL)
            print("Loaded custom model for \(language.displayName)")
        } catch {
            print("Failed to load custom model: \(error)")
        }
    }
    
    // MARK: - Confidence Scoring
    
    private func calculateConfidence(for features: [Float]) -> Float {
        // Calculate confidence score for recognition
        // This is a simplified implementation
        
        let energy = features.map { $0 * $0 }.reduce(0, +)
        let normalizedEnergy = energy / Float(features.count)
        
        // Simple confidence based on energy
        return min(normalizedEnergy * 10.0, 1.0)
    }
}

// MARK: - SFSpeechRecognizerDelegate

extension SpeechRecognizer: SFSpeechRecognizerDelegate {
    func speechRecognizer(_ speechRecognizer: SFSpeechRecognizer, availabilityDidChange available: Bool) {
        if !available {
            delegate?.speechRecognizer(self, didEncounterError: SpeechRecognizerError.speechRecognizerNotAvailable)
        }
    }
}

// MARK: - Error Types

enum SpeechRecognizerError: Error, LocalizedError {
    case audioEngineCreationFailed
    case audioEngineNotInitialized
    case recognitionRequestCreationFailed
    case speechRecognizerNotAvailable
    case modelLoadingFailed
    case featureExtractionFailed
    
    var errorDescription: String? {
        switch self {
        case .audioEngineCreationFailed:
            return "Failed to create audio engine"
        case .audioEngineNotInitialized:
            return "Audio engine not initialized"
        case .recognitionRequestCreationFailed:
            return "Failed to create recognition request"
        case .speechRecognizerNotAvailable:
            return "Speech recognizer not available for current language"
        case .modelLoadingFailed:
            return "Failed to load custom recognition model"
        case .featureExtractionFailed:
            return "Failed to extract audio features"
        }
    }
} 