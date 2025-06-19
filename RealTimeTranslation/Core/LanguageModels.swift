import Foundation
import Speech

enum Language: String, CaseIterable {
    case english = "en"
    case spanish = "es"
    case french = "fr"
    case german = "de"
    case italian = "it"
    case portuguese = "pt"
    case chinese = "zh"
    case japanese = "ja"
    case korean = "ko"
    case arabic = "ar"
    case russian = "ru"
    case hindi = "hi"
    
    var displayName: String {
        switch self {
        case .english: return "English"
        case .spanish: return "Español"
        case .french: return "Français"
        case .german: return "Deutsch"
        case .italian: return "Italiano"
        case .portuguese: return "Português"
        case .chinese: return "中文"
        case .japanese: return "日本語"
        case .korean: return "한국어"
        case .arabic: return "العربية"
        case .russian: return "Русский"
        case .hindi: return "हिन्दी"
        }
    }
    
    var speechRecognitionLanguage: String {
        switch self {
        case .english: return "en-US"
        case .spanish: return "es-ES"
        case .french: return "fr-FR"
        case .german: return "de-DE"
        case .italian: return "it-IT"
        case .portuguese: return "pt-BR"
        case .chinese: return "zh-CN"
        case .japanese: return "ja-JP"
        case .korean: return "ko-KR"
        case .arabic: return "ar-SA"
        case .russian: return "ru-RU"
        case .hindi: return "hi-IN"
        }
    }
    
    var speechSynthesisVoice: String {
        switch self {
        case .english: return "com.apple.ttsbundle.siri_female_en-US_compact"
        case .spanish: return "com.apple.ttsbundle.Monica-compact"
        case .french: return "com.apple.ttsbundle.Aurelie-compact"
        case .german: return "com.apple.ttsbundle.Anna-compact"
        case .italian: return "com.apple.ttsbundle.Alice-compact"
        case .portuguese: return "com.apple.ttsbundle.Luciana-compact"
        case .chinese: return "com.apple.ttsbundle.Ting-Ting-compact"
        case .japanese: return "com.apple.ttsbundle.Kyoko-compact"
        case .korean: return "com.apple.ttsbundle.Yuna-compact"
        case .arabic: return "com.apple.ttsbundle.Tarik-compact"
        case .russian: return "com.apple.ttsbundle.Yuri-compact"
        case .hindi: return "com.apple.ttsbundle.Lekha-compact"
        }
    }
}

struct TranslationModel {
    let sourceLanguage: Language
    let targetLanguage: Language
    let modelName: String
    let modelURL: URL?
    
    init(sourceLanguage: Language, targetLanguage: Language) {
        self.sourceLanguage = sourceLanguage
        self.targetLanguage = targetLanguage
        self.modelName = "\(sourceLanguage.rawValue)_to_\(targetLanguage.rawValue)_transformer"
        
        // In a real app, this would point to your trained Core ML model
        self.modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc")
    }
    
    var isAvailable: Bool {
        return modelURL != nil
    }
}

struct SpeechRecognitionModel {
    let language: Language
    let modelName: String
    let modelURL: URL?
    
    init(language: Language) {
        self.language = language
        self.modelName = "\(language.rawValue)_speech_recognition"
        
        // In a real app, this would point to your trained speech recognition model
        self.modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc")
    }
    
    var isAvailable: Bool {
        return modelURL != nil
    }
}

// Model configuration for different device capabilities
enum ModelConfiguration {
    case small    // For older devices
    case medium   // For most devices
    case large    // For high-end devices
    
    var maxSequenceLength: Int {
        switch self {
        case .small: return 128
        case .medium: return 256
        case .large: return 512
        }
    }
    
    var modelSize: String {
        switch self {
        case .small: return "small"
        case .medium: return "medium"
        case .large: return "large"
        }
    }
}

// Audio configuration
struct AudioConfiguration {
    let sampleRate: Double = 16000
    let channels: Int = 1
    let bitDepth: Int = 16
    let bufferSize: Int = 1024
    
    var audioFormat: AVAudioFormat? {
        return AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: sampleRate,
            channels: AVAudioChannelCount(channels),
            interleaved: true
        )
    }
} 