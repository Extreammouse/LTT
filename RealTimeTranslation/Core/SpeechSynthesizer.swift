import Foundation
import AVFoundation
import Speech

protocol SpeechSynthesizerDelegate: AnyObject {
    func speechSynthesizer(_ synthesizer: SpeechSynthesizer, didStartSpeaking text: String)
    func speechSynthesizer(_ synthesizer: SpeechSynthesizer, didFinishSpeaking text: String)
    func speechSynthesizer(_ synthesizer: SpeechSynthesizer, didEncounterError error: Error)
}

class SpeechSynthesizer: NSObject {
    weak var delegate: SpeechSynthesizerDelegate?
    
    private var synthesizer: AVSpeechSynthesizer?
    private var currentUtterance: AVSpeechUtterance?
    private var currentLanguage: Language = .english
    private var isSpeaking = false
    
    private let processingQueue = DispatchQueue(label: "speech.synthesis", qos: .userInteractive)
    
    override init() {
        super.init()
        setupSynthesizer()
    }
    
    deinit {
        stopSpeaking()
    }
    
    // MARK: - Public Methods
    
    func speak(text: String, language: Language) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        
        currentLanguage = language
        
        processingQueue.async { [weak self] in
            self?.performSpeechSynthesis(text: text)
        }
    }
    
    func stopSpeaking() {
        guard isSpeaking else { return }
        
        synthesizer?.stopSpeaking(at: .immediate)
        currentUtterance = nil
        isSpeaking = false
    }
    
    func pauseSpeaking() {
        guard isSpeaking else { return }
        
        synthesizer?.pauseSpeaking(at: .immediate)
    }
    
    func resumeSpeaking() {
        synthesizer?.continueSpeaking()
    }
    
    // MARK: - Private Methods
    
    private func setupSynthesizer() {
        synthesizer = AVSpeechSynthesizer()
        synthesizer?.delegate = self
    }
    
    private func performSpeechSynthesis(text: String) {
        do {
            // Preprocess text for speech synthesis
            let preprocessedText = preprocessTextForSpeech(text)
            
            // Create utterance
            let utterance = createUtterance(text: preprocessedText, language: currentLanguage)
            
            // Start speaking
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                
                self.currentUtterance = utterance
                self.synthesizer?.speak(utterance)
                self.isSpeaking = true
                self.delegate?.speechSynthesizer(self, didStartSpeaking: text)
            }
        } catch {
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.delegate?.speechSynthesizer(self, didEncounterError: error)
            }
        }
    }
    
    private func preprocessTextForSpeech(_ text: String) -> String {
        var processedText = text
        
        // Normalize whitespace
        processedText = processedText.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        
        // Remove extra punctuation that might interfere with speech
        processedText = processedText.replacingOccurrences(of: "[\\[\\]{}()]", with: "", options: .regularExpression)
        
        // Handle language-specific preprocessing
        processedText = handleLanguageSpecificPreprocessing(processedText, for: currentLanguage)
        
        return processedText.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    
    private func handleLanguageSpecificPreprocessing(_ text: String, for language: Language) -> String {
        switch language {
        case .spanish:
            // Handle Spanish-specific text preprocessing
            return text.replacingOccurrences(of: "¿", with: "")
                       .replacingOccurrences(of: "¡", with: "")
        case .french:
            // Handle French-specific text preprocessing
            return text
        case .german:
            // Handle German-specific text preprocessing
            return text
        case .chinese:
            // Handle Chinese-specific text preprocessing
            return text
        case .japanese:
            // Handle Japanese-specific text preprocessing
            return text
        default:
            return text
        }
    }
    
    private func createUtterance(text: String, language: Language) -> AVSpeechUtterance {
        let utterance = AVSpeechUtterance(string: text)
        
        // Set voice
        utterance.voice = selectVoice(for: language)
        
        // Set speech rate (words per minute)
        utterance.rate = selectSpeechRate(for: language)
        
        // Set pitch
        utterance.pitchMultiplier = selectPitch(for: language)
        
        // Set volume
        utterance.volume = 0.8
        
        // Set pre-utterance delay
        utterance.preUtteranceDelay = 0.1
        
        // Set post-utterance delay
        utterance.postUtteranceDelay = 0.2
        
        return utterance
    }
    
    private func selectVoice(for language: Language) -> AVSpeechSynthesisVoice? {
        // Try to get the specific voice for the language
        if let voice = AVSpeechSynthesisVoice(identifier: language.speechSynthesisVoice) {
            return voice
        }
        
        // Fallback to any available voice for the language
        let locale = Locale(identifier: language.speechRecognitionLanguage)
        return AVSpeechSynthesisVoice(language: locale.languageCode)
    }
    
    private func selectSpeechRate(for language: Language) -> Float {
        // Adjust speech rate based on language characteristics
        switch language {
        case .spanish:
            return 0.5 // Spanish tends to be spoken faster
        case .french:
            return 0.45 // French has a moderate pace
        case .german:
            return 0.4 // German can be slower due to compound words
        case .chinese:
            return 0.4 // Chinese tones require slower speech
        case .japanese:
            return 0.4 // Japanese syllables are distinct
        default:
            return 0.5 // Default rate
        }
    }
    
    private func selectPitch(for language: Language) -> Float {
        // Adjust pitch based on language characteristics
        switch language {
        case .spanish:
            return 1.1 // Spanish can have higher pitch variations
        case .french:
            return 1.0 // French has moderate pitch
        case .german:
            return 0.9 // German can have lower pitch
        case .chinese:
            return 1.0 // Chinese tones affect pitch perception
        case .japanese:
            return 1.0 // Japanese has moderate pitch
        default:
            return 1.0 // Default pitch
        }
    }
    
    // MARK: - Advanced Features
    
    func synthesizeWithCustomVoice(text: String, voiceSettings: VoiceSettings) {
        let utterance = AVSpeechUtterance(string: text)
        
        // Apply custom voice settings
        utterance.voice = voiceSettings.voice
        utterance.rate = voiceSettings.rate
        utterance.pitchMultiplier = voiceSettings.pitch
        utterance.volume = voiceSettings.volume
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            self.currentUtterance = utterance
            self.synthesizer?.speak(utterance)
            self.isSpeaking = true
            self.delegate?.speechSynthesizer(self, didStartSpeaking: text)
        }
    }
    
    func synthesizeWithEmotion(text: String, emotion: SpeechEmotion) {
        let utterance = AVSpeechUtterance(string: text)
        
        // Apply emotion-specific settings
        switch emotion {
        case .happy:
            utterance.pitchMultiplier = 1.2
            utterance.rate = 0.6
        case .sad:
            utterance.pitchMultiplier = 0.8
            utterance.rate = 0.3
        case .angry:
            utterance.pitchMultiplier = 1.3
            utterance.rate = 0.7
        case .calm:
            utterance.pitchMultiplier = 0.9
            utterance.rate = 0.4
        }
        
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            self.currentUtterance = utterance
            self.synthesizer?.speak(utterance)
            self.isSpeaking = true
            self.delegate?.speechSynthesizer(self, didStartSpeaking: text)
        }
    }
    
    // MARK: - Audio Processing
    
    func processAudioForBetterQuality(_ utterance: AVSpeechUtterance) {
        // Apply audio processing for better quality
        // This could include noise reduction, equalization, etc.
        
        // For now, we'll use the default settings
        // In a real implementation, you might:
        // 1. Apply audio filters
        // 2. Adjust frequency response
        // 3. Add reverb or other effects
    }
}

// MARK: - AVSpeechSynthesizerDelegate

extension SpeechSynthesizer: AVSpeechSynthesizerDelegate {
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        // Speech started
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            self.isSpeaking = false
            self.currentUtterance = nil
            
            if let text = utterance.speechString {
                self.delegate?.speechSynthesizer(self, didFinishSpeaking: text)
            }
        }
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didPause utterance: AVSpeechUtterance) {
        // Speech paused
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didContinue utterance: AVSpeechUtterance) {
        // Speech resumed
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            self.isSpeaking = false
            self.currentUtterance = nil
        }
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, willSpeakRangeOfSpeechString characterRange: NSRange, utterance: AVSpeechUtterance) {
        // About to speak a specific range of text
    }
}

// MARK: - Supporting Types

struct VoiceSettings {
    let voice: AVSpeechSynthesisVoice?
    let rate: Float
    let pitch: Float
    let volume: Float
    
    init(voice: AVSpeechSynthesisVoice? = nil, rate: Float = 0.5, pitch: Float = 1.0, volume: Float = 0.8) {
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.volume = volume
    }
}

enum SpeechEmotion {
    case happy
    case sad
    case angry
    case calm
}

// MARK: - Error Types

enum SpeechSynthesizerError: Error, LocalizedError {
    case synthesizerNotAvailable
    case voiceNotAvailable
    case textProcessingFailed
    case audioOutputFailed
    
    var errorDescription: String? {
        switch self {
        case .synthesizerNotAvailable:
            return "Speech synthesizer not available"
        case .voiceNotAvailable:
            return "Voice not available for selected language"
        case .textProcessingFailed:
            return "Failed to process text for speech synthesis"
        case .audioOutputFailed:
            return "Failed to output audio"
        }
    }
} 