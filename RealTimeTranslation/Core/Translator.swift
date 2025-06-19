import Foundation
import CoreML
import NaturalLanguage

protocol TranslatorDelegate: AnyObject {
    func translator(_ translator: Translator, didTranslateText text: String, from sourceLanguage: Language, to targetLanguage: Language)
    func translator(_ translator: Translator, didEncounterError error: Error)
}

class Translator {
    weak var delegate: TranslatorDelegate?
    
    private var translationModel: MLModel?
    private var tokenizer: NLTokenizer?
    private var currentSourceLanguage: Language = .english
    private var currentTargetLanguage: Language = .english
    
    private let processingQueue = DispatchQueue(label: "translation.processing", qos: .userInteractive)
    private let modelConfiguration = ModelConfiguration.medium
    
    // MARK: - Public Methods
    
    func translate(text: String, from sourceLanguage: Language, to targetLanguage: Language) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        
        currentSourceLanguage = sourceLanguage
        currentTargetLanguage = targetLanguage
        
        processingQueue.async { [weak self] in
            self?.performTranslation(text: text)
        }
    }
    
    func loadModel(for sourceLanguage: Language, targetLanguage: Language) {
        let model = TranslationModel(sourceLanguage: sourceLanguage, targetLanguage: targetLanguage)
        
        guard let modelURL = model.modelURL else {
            print("Translation model not available for \(sourceLanguage.displayName) to \(targetLanguage.displayName)")
            return
        }
        
        do {
            translationModel = try MLModel(contentsOf: modelURL)
            setupTokenizer(for: sourceLanguage)
            print("Loaded translation model for \(sourceLanguage.displayName) to \(targetLanguage.displayName)")
        } catch {
            print("Failed to load translation model: \(error)")
            delegate?.translator(self, didEncounterError: TranslatorError.modelLoadingFailed)
        }
    }
    
    // MARK: - Private Methods
    
    private func performTranslation(text: String) {
        do {
            // Preprocess text
            let preprocessedText = preprocessText(text)
            
            // Tokenize input
            let tokens = tokenizeText(preprocessedText)
            
            // Perform translation
            let translatedTokens = try translateTokens(tokens)
            
            // Post-process and detokenize
            let translatedText = detokenizeText(translatedTokens)
            
            // Notify delegate
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.delegate?.translator(
                    self,
                    didTranslateText: translatedText,
                    from: self.currentSourceLanguage,
                    to: self.currentTargetLanguage
                )
            }
        } catch {
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.delegate?.translator(self, didEncounterError: error)
            }
        }
    }
    
    private func preprocessText(_ text: String) -> String {
        var processedText = text
        
        // Normalize whitespace
        processedText = processedText.replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        
        // Remove extra punctuation
        processedText = processedText.trimmingCharacters(in: .whitespacesAndNewlines)
        
        // Handle special characters based on language
        processedText = handleSpecialCharacters(processedText, for: currentSourceLanguage)
        
        return processedText
    }
    
    private func handleSpecialCharacters(_ text: String, for language: Language) -> String {
        switch language {
        case .spanish:
            // Handle Spanish-specific characters
            return text.replacingOccurrences(of: "¿", with: "")
                       .replacingOccurrences(of: "¡", with: "")
        case .french:
            // Handle French-specific characters
            return text
        case .german:
            // Handle German-specific characters
            return text
        default:
            return text
        }
    }
    
    private func setupTokenizer(for language: Language) {
        let locale = Locale(identifier: language.speechRecognitionLanguage)
        tokenizer = NLTokenizer(unit: .word)
        tokenizer?.setLanguage(locale.languageCode ?? "en")
    }
    
    private func tokenizeText(_ text: String) -> [String] {
        guard let tokenizer = tokenizer else {
            return text.components(separatedBy: .whitespaces)
        }
        
        let range = NSRange(location: 0, length: text.utf16.count)
        let tokens = tokenizer.tokens(for: range)
        
        return tokens.map { String(text[Range($0, in: text)!]) }
    }
    
    private func translateTokens(_ tokens: [String]) throws -> [String] {
        // If we have a custom model, use it
        if let model = translationModel {
            return try translateWithCustomModel(tokens, model: model)
        } else {
            // Fallback to rule-based translation
            return translateWithRules(tokens)
        }
    }
    
    private func translateWithCustomModel(_ tokens: [String], model: MLModel) throws -> [String] {
        // This is where you'd use your trained transformer model
        // For now, we'll implement a simplified version
        
        let inputFeatures = prepareInputFeatures(from: tokens)
        
        // Create MLFeatureProvider for model input
        let input = try createModelInput(features: inputFeatures)
        
        // Run inference
        let output = try model.prediction(from: input)
        
        // Extract output features
        let outputFeatures = extractOutputFeatures(from: output)
        
        // Convert features back to tokens
        return convertFeaturesToTokens(outputFeatures)
    }
    
    private func prepareInputFeatures(from tokens: [String]) -> [Float] {
        // Convert tokens to numerical features
        // This is a simplified implementation
        
        var features: [Float] = []
        let maxTokens = min(tokens.count, modelConfiguration.maxSequenceLength)
        
        for i in 0..<maxTokens {
            let token = tokens[i]
            let tokenFeatures = tokenToFeatures(token)
            features.append(contentsOf: tokenFeatures)
        }
        
        // Pad if necessary
        while features.count < modelConfiguration.maxSequenceLength * 128 { // Assuming 128 features per token
            features.append(0.0)
        }
        
        return features
    }
    
    private func tokenToFeatures(_ token: String) -> [Float] {
        // Convert token to numerical features
        // In practice, you'd use a proper tokenizer like BPE or WordPiece
        
        var features = [Float](repeating: 0.0, count: 128)
        
        // Simple character-based encoding
        for (index, char) in token.enumerated() {
            if index < 128 {
                features[index] = Float(char.asciiValue ?? 0) / 255.0
            }
        }
        
        return features
    }
    
    private func createModelInput(features: [Float]) throws -> MLFeatureProvider {
        // Create MLFeatureProvider for model input
        // This is a simplified implementation
        
        let inputArray = try MLMultiArray(shape: [1, NSNumber(value: features.count)], dataType: .float32)
        
        for (index, feature) in features.enumerated() {
            inputArray[index] = NSNumber(value: feature)
        }
        
        let input = try MLDictionaryFeatureProvider(dictionary: ["input": inputArray])
        return input
    }
    
    private func extractOutputFeatures(from output: MLFeatureProvider) -> [Float] {
        // Extract features from model output
        // This is a simplified implementation
        
        guard let outputArray = output.featureValue(for: "output")?.multiArrayValue else {
            return []
        }
        
        var features: [Float] = []
        let count = outputArray.count
        
        for i in 0..<count {
            features.append(outputArray[i].floatValue)
        }
        
        return features
    }
    
    private func convertFeaturesToTokens(_ features: [Float]) -> [String] {
        // Convert numerical features back to tokens
        // This is a simplified implementation
        
        var tokens: [String] = []
        let tokensPerFeature = 128 // Assuming 128 features per token
        
        for i in stride(from: 0, to: features.count, by: tokensPerFeature) {
            let tokenFeatures = Array(features[i..<min(i + tokensPerFeature, features.count)])
            let token = featuresToToken(tokenFeatures)
            if !token.isEmpty {
                tokens.append(token)
            }
        }
        
        return tokens
    }
    
    private func featuresToToken(_ features: [Float]) -> String {
        // Convert features back to token string
        // This is a simplified implementation
        
        var token = ""
        
        for feature in features {
            if feature > 0.1 { // Threshold for character detection
                let asciiValue = Int(feature * 255)
                if let char = Character(UnicodeScalar(asciiValue)!) {
                    token.append(char)
                }
            }
        }
        
        return token
    }
    
    private func translateWithRules(_ tokens: [String]) -> [String] {
        // Rule-based translation as fallback
        // This is a very simplified implementation
        
        var translatedTokens: [String] = []
        
        for token in tokens {
            let translatedToken = translateTokenWithRules(token)
            translatedTokens.append(translatedToken)
        }
        
        return translatedTokens
    }
    
    private func translateTokenWithRules(_ token: String) -> String {
        // Simple dictionary-based translation
        // In practice, you'd use a comprehensive dictionary
        
        let lowercasedToken = token.lowercased()
        
        switch (currentSourceLanguage, currentTargetLanguage) {
        case (.spanish, .english):
            return spanishToEnglishDictionary[lowercasedToken] ?? token
        case (.english, .spanish):
            return englishToSpanishDictionary[lowercasedToken] ?? token
        default:
            return token
        }
    }
    
    private func detokenizeText(_ tokens: [String]) -> String {
        // Combine tokens back into text
        return tokens.joined(separator: " ")
    }
    
    // MARK: - Translation Dictionaries
    
    private let spanishToEnglishDictionary: [String: String] = [
        "hola": "hello",
        "adiós": "goodbye",
        "gracias": "thank you",
        "por favor": "please",
        "sí": "yes",
        "no": "no",
        "buenos días": "good morning",
        "buenas noches": "good night",
        "cómo estás": "how are you",
        "bien": "good",
        "mal": "bad",
        "agua": "water",
        "comida": "food",
        "casa": "house",
        "coche": "car",
        "trabajo": "work",
        "tiempo": "time",
        "persona": "person",
        "familia": "family",
        "amigo": "friend"
    ]
    
    private let englishToSpanishDictionary: [String: String] = [
        "hello": "hola",
        "goodbye": "adiós",
        "thank you": "gracias",
        "please": "por favor",
        "yes": "sí",
        "no": "no",
        "good morning": "buenos días",
        "good night": "buenas noches",
        "how are you": "cómo estás",
        "good": "bien",
        "bad": "mal",
        "water": "agua",
        "food": "comida",
        "house": "casa",
        "car": "coche",
        "work": "trabajo",
        "time": "tiempo",
        "person": "persona",
        "family": "familia",
        "friend": "amigo"
    ]
}

// MARK: - Error Types

enum TranslatorError: Error, LocalizedError {
    case modelLoadingFailed
    case translationFailed
    case invalidInput
    case tokenizationFailed
    
    var errorDescription: String? {
        switch self {
        case .modelLoadingFailed:
            return "Failed to load translation model"
        case .translationFailed:
            return "Translation failed"
        case .invalidInput:
            return "Invalid input text"
        case .tokenizationFailed:
            return "Failed to tokenize text"
        }
    }
} 