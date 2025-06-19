import Foundation
import AVFoundation
import Accelerate

protocol AudioProcessorDelegate: AnyObject {
    func audioProcessor(_ processor: AudioProcessor, didReceiveAudioBuffer buffer: [Float])
    func audioProcessor(_ processor: AudioProcessor, didEncounterError error: Error)
}

class AudioProcessor: NSObject {
    weak var delegate: AudioProcessorDelegate?
    
    private var audioEngine: AVAudioEngine?
    private var inputNode: AVAudioInputNode?
    private var audioFormat: AVAudioFormat?
    private var isRecording = false
    
    private let audioConfiguration = AudioConfiguration()
    private let processingQueue = DispatchQueue(label: "audio.processing", qos: .userInteractive)
    
    override init() {
        super.init()
        setupAudioSession()
    }
    
    deinit {
        stopRecording()
    }
    
    // MARK: - Public Methods
    
    func startRecording() throws {
        guard !isRecording else { return }
        
        do {
            try setupAudioEngine()
            try startAudioEngine()
            isRecording = true
        } catch {
            delegate?.audioProcessor(self, didEncounterError: error)
            throw error
        }
    }
    
    func stopRecording() {
        guard isRecording else { return }
        
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        isRecording = false
    }
    
    // MARK: - Private Methods
    
    private func setupAudioSession() {
        do {
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print("Failed to setup audio session: \(error)")
        }
    }
    
    private func setupAudioEngine() throws {
        audioEngine = AVAudioEngine()
        guard let audioEngine = audioEngine else {
            throw AudioProcessorError.engineCreationFailed
        }
        
        inputNode = audioEngine.inputNode
        guard let inputNode = inputNode else {
            throw AudioProcessorError.inputNodeNotFound
        }
        
        audioFormat = audioConfiguration.audioFormat
        guard let audioFormat = audioFormat else {
            throw AudioProcessorError.invalidAudioFormat
        }
        
        // Install tap on input node
        inputNode.installTap(onBus: 0, bufferSize: AVAudioFrameCount(audioConfiguration.bufferSize), format: audioFormat) { [weak self] buffer, _ in
            self?.processAudioBuffer(buffer)
        }
    }
    
    private func startAudioEngine() throws {
        guard let audioEngine = audioEngine else {
            throw AudioProcessorError.engineNotInitialized
        }
        
        audioEngine.prepare()
        try audioEngine.start()
    }
    
    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        processingQueue.async { [weak self] in
            guard let self = self else { return }
            
            do {
                let audioData = try self.extractAudioData(from: buffer)
                let processedData = self.preprocessAudioData(audioData)
                self.delegate?.audioProcessor(self, didReceiveAudioBuffer: processedData)
            } catch {
                self.delegate?.audioProcessor(self, didEncounterError: error)
            }
        }
    }
    
    private func extractAudioData(from buffer: AVAudioPCMBuffer) throws -> [Float] {
        guard let channelData = buffer.floatChannelData?[0] else {
            throw AudioProcessorError.invalidBufferData
        }
        
        let frameLength = Int(buffer.frameLength)
        let audioData = Array(UnsafeBufferPointer(start: channelData, count: frameLength))
        
        return audioData
    }
    
    private func preprocessAudioData(_ audioData: [Float]) -> [Float] {
        // Apply noise reduction and normalization
        var processedData = audioData
        
        // Normalize audio levels
        processedData = normalizeAudio(processedData)
        
        // Apply simple noise gate
        processedData = applyNoiseGate(processedData, threshold: 0.01)
        
        // Apply high-pass filter to remove low-frequency noise
        processedData = applyHighPassFilter(processedData)
        
        return processedData
    }
    
    private func normalizeAudio(_ audioData: [Float]) -> [Float] {
        guard !audioData.isEmpty else { return audioData }
        
        // Find the maximum absolute value
        let maxValue = audioData.map { abs($0) }.max() ?? 1.0
        
        // Avoid division by zero
        guard maxValue > 0 else { return audioData }
        
        // Normalize to [-1, 1] range
        let normalizedData = audioData.map { $0 / maxValue }
        
        return normalizedData
    }
    
    private func applyNoiseGate(_ audioData: [Float], threshold: Float) -> [Float] {
        return audioData.map { sample in
            return abs(sample) < threshold ? 0.0 : sample
        }
    }
    
    private func applyHighPassFilter(_ audioData: [Float]) -> [Float] {
        // Simple high-pass filter implementation
        var filteredData = [Float](repeating: 0.0, count: audioData.count)
        let alpha: Float = 0.95 // Filter coefficient
        
        if audioData.count > 0 {
            filteredData[0] = audioData[0]
            
            for i in 1..<audioData.count {
                filteredData[i] = alpha * (filteredData[i-1] + audioData[i] - audioData[i-1])
            }
        }
        
        return filteredData
    }
    
    // MARK: - Audio Feature Extraction
    
    func extractMFCCFeatures(from audioData: [Float]) -> [[Float]] {
        // This is a simplified MFCC implementation
        // In a real app, you'd use a more sophisticated library like Accelerate framework
        
        let frameSize = 512
        let hopSize = 256
        let numFrames = (audioData.count - frameSize) / hopSize + 1
        
        var mfccFeatures: [[Float]] = []
        
        for frameIndex in 0..<numFrames {
            let startIndex = frameIndex * hopSize
            let endIndex = min(startIndex + frameSize, audioData.count)
            
            let frame = Array(audioData[startIndex..<endIndex])
            let mfcc = computeMFCC(for: frame)
            mfccFeatures.append(mfcc)
        }
        
        return mfccFeatures
    }
    
    private func computeMFCC(for frame: [Float]) -> [Float] {
        // Simplified MFCC computation
        // In practice, you'd use FFT and mel-filterbank
        
        let numCoefficients = 13
        var mfcc = [Float](repeating: 0.0, count: numCoefficients)
        
        // Simple energy-based features as placeholder
        let energy = frame.map { $0 * $0 }.reduce(0, +)
        let logEnergy = log(max(energy, 1e-10))
        
        for i in 0..<numCoefficients {
            mfcc[i] = logEnergy * Float(i + 1) / Float(numCoefficients)
        }
        
        return mfcc
    }
}

// MARK: - Error Types

enum AudioProcessorError: Error, LocalizedError {
    case engineCreationFailed
    case inputNodeNotFound
    case invalidAudioFormat
    case engineNotInitialized
    case invalidBufferData
    case permissionDenied
    
    var errorDescription: String? {
        switch self {
        case .engineCreationFailed:
            return "Failed to create audio engine"
        case .inputNodeNotFound:
            return "Audio input node not found"
        case .invalidAudioFormat:
            return "Invalid audio format"
        case .engineNotInitialized:
            return "Audio engine not initialized"
        case .invalidBufferData:
            return "Invalid audio buffer data"
        case .permissionDenied:
            return "Microphone permission denied"
        }
    }
} 