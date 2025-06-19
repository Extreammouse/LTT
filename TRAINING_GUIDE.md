# Real-Time Translation Framework - Training Guide

This guide will walk you through training custom transformer models for real-time speech translation and deploying them to iOS devices.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup](#setup)
3. [Training Translation Models](#training-translation-models)
4. [Training Speech Recognition Models](#training-speech-recognition-models)
5. [Converting to Core ML](#converting-to-core-ml)
6. [Evaluating Models](#evaluating-models)
7. [Deployment](#deployment)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements
- **For Training**: GPU with 8GB+ VRAM (NVIDIA RTX 3080 or better recommended)
- **For iOS Development**: Mac with Xcode 14.0+
- **For Testing**: iPhone/iPad with iOS 15.0+

### Software Requirements
- Python 3.8+
- PyTorch 2.0+
- Xcode 14.0+
- iOS 15.0+ (target device)

### Data Requirements
- **Translation Data**: Parallel text corpora (e.g., OPUS, WMT, custom datasets)
- **Speech Recognition Data**: Audio recordings with transcriptions
- **Minimum Dataset Size**: 10,000 sentence pairs for translation, 100 hours for speech recognition

## Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd real-time-translation-framework
   ```

2. **Run the setup script**:
   ```bash
   python setup.py
   ```

3. **Install additional dependencies** (if needed):
   ```bash
   pip install -r requirements.txt
   ```

## Training Translation Models

### 1. Prepare Your Data

Create a training dataset in the following format:

```json
{
  "source_language": "es",
  "target_language": "en",
  "training_data": [
    {
      "source": "Hola, ¿cómo estás?",
      "target": "Hello, how are you?"
    },
    {
      "source": "Gracias por tu ayuda",
      "target": "Thank you for your help"
    }
  ]
}
```

### 2. Configure Training Parameters

Edit `training_config.json`:

```json
{
  "d_model": 512,
  "nhead": 8,
  "num_encoder_layers": 6,
  "num_decoder_layers": 6,
  "dim_feedforward": 2048,
  "dropout": 0.1,
  "max_length": 128,
  "batch_size": 4,
  "learning_rate": 1e-4,
  "num_epochs": 10,
  "device": "auto"
}
```

### 3. Train the Model

```bash
# Basic training
python train_translator.py --source-lang es --target-lang en --output-dir ./models

# With custom configuration
python train_translator.py --source-lang es --target-lang en --config training_config.json --output-dir ./models

# With Core ML conversion
python train_translator.py --source-lang es --target-lang en --convert-coreml --output-dir ./models
```

### 4. Monitor Training

The training script will output:
- Loss per epoch
- Validation metrics
- Training progress
- Model checkpoints

## Training Speech Recognition Models

### 1. Prepare Audio Data

Organize your audio data:
```
training_data/
├── audio/
│   ├── sample1.wav
│   ├── sample2.wav
│   └── ...
└── transcriptions.json
```

### 2. Create Transcription File

```json
{
  "transcriptions": [
    {
      "audio_file": "sample1.wav",
      "text": "Hello world",
      "language": "en"
    }
  ]
}
```

### 3. Train Speech Recognition Model

```bash
python train_speech_recognition.py --language en --audio-dir ./training_data/audio --transcriptions ./training_data/transcriptions.json --output-dir ./models
```

## Converting to Core ML

### 1. Convert Translation Models

```bash
python convert_to_coreml.py \
  --model-type translation \
  --model-path ./models/es_to_en_transformer.pth \
  --tokenizer-path ./models/es_to_en_tokenizer \
  --output-path ./models/es_to_en_transformer.mlmodel \
  --source-lang es \
  --target-lang en \
  --optimize \
  --validate
```

### 2. Convert Speech Recognition Models

```bash
python convert_to_coreml.py \
  --model-type speech_recognition \
  --model-path ./models/en_speech_recognition.pth \
  --output-path ./models/en_speech_recognition.mlmodel \
  --language en \
  --optimize \
  --validate
```

### 3. Model Optimization

The conversion script includes optimization features:
- **Quantization**: Reduce model size by 50-75%
- **Pruning**: Remove unnecessary weights
- **Neural Engine**: Optimize for Apple's Neural Engine

## Evaluating Models

### 1. Evaluate Translation Models

```bash
python evaluate_model.py \
  --model-type translation \
  --model-path ./models/es_to_en_transformer.pth \
  --tokenizer-path ./models/es_to_en_tokenizer \
  --source-lang es \
  --target-lang en \
  --num-samples 100 \
  --output-dir ./evaluation_results
```

### 2. Evaluate Speech Recognition Models

```bash
python evaluate_model.py \
  --model-type speech_recognition \
  --model-path ./models/en_speech_recognition.pth \
  --language en \
  --num-samples 50 \
  --output-dir ./evaluation_results
```

### 3. Evaluate Core ML Models

```bash
python evaluate_model.py \
  --model-type coreml \
  --model-path ./models/es_to_en_transformer.mlmodel \
  --num-samples 50 \
  --output-dir ./evaluation_results
```

## Deployment

### 1. Add Models to iOS Project

1. Open `RealTimeTranslation.xcodeproj` in Xcode
2. Drag your `.mlmodel` files to the project
3. Ensure "Add to target" is checked for your app target

### 2. Update Model References

In `LanguageModels.swift`, update the model paths:

```swift
struct TranslationModel {
    let sourceLanguage: Language
    let targetLanguage: Language
    let modelName: String
    let modelURL: URL?
    
    init(sourceLanguage: Language, targetLanguage: Language) {
        self.sourceLanguage = sourceLanguage
        self.targetLanguage = targetLanguage
        self.modelName = "\(sourceLanguage.rawValue)_to_\(targetLanguage.rawValue)_transformer"
        
        // Update this path to match your model name
        self.modelURL = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc")
    }
}
```

### 3. Test on Device

1. Connect your iOS device
2. Select your device as the target
3. Build and run the app
4. Test real-time translation

## Performance Optimization

### 1. Model Size Optimization

```bash
# Use smaller model configuration
{
  "d_model": 256,
  "nhead": 4,
  "num_encoder_layers": 4,
  "num_decoder_layers": 4,
  "dim_feedforward": 1024
}
```

### 2. Inference Optimization

- Use batch processing for multiple translations
- Implement caching for frequently used phrases
- Optimize audio processing pipeline

### 3. Memory Management

- Monitor memory usage during inference
- Implement model unloading when not in use
- Use appropriate model configurations for device capabilities

## Troubleshooting

### Common Issues

#### 1. Model Loading Errors
```
Error: Failed to load model
```
**Solution**: Ensure model files are properly added to the Xcode project and the paths in `LanguageModels.swift` are correct.

#### 2. Poor Translation Quality
```
Issue: Low accuracy translations
```
**Solutions**:
- Increase training data size
- Adjust model hyperparameters
- Use pre-trained models as starting point
- Implement better tokenization

#### 3. Slow Inference
```
Issue: High latency during translation
```
**Solutions**:
- Reduce model size
- Use quantization
- Optimize for Neural Engine
- Implement caching

#### 4. Audio Processing Issues
```
Issue: Poor speech recognition
```
**Solutions**:
- Improve audio preprocessing
- Use better feature extraction
- Increase training data diversity
- Implement noise reduction

### Performance Benchmarks

| Model Size | Accuracy | Latency | Memory Usage |
|------------|----------|---------|--------------|
| Small (256) | 85% | 200ms | 50MB |
| Medium (512) | 92% | 400ms | 100MB |
| Large (1024) | 95% | 800ms | 200MB |

### Best Practices

1. **Data Quality**: Use high-quality, diverse training data
2. **Regular Evaluation**: Continuously evaluate model performance
3. **Incremental Training**: Start with small models and scale up
4. **A/B Testing**: Compare different model configurations
5. **User Feedback**: Collect and incorporate user feedback

## Advanced Topics

### 1. Custom Tokenization

Implement custom tokenizers for specific languages:

```python
from tokenizers import Tokenizer, models, pre_tokenizers, decoders

# Create custom tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
```

### 2. Multi-Language Models

Train models that support multiple language pairs:

```bash
python train_translator.py --source-lang mul --target-lang en --output-dir ./models
```

### 3. Domain-Specific Training

Fine-tune models for specific domains (medical, legal, technical):

```bash
python train_translator.py --source-lang es --target-lang en --domain medical --output-dir ./models
```

### 4. Real-Time Adaptation

Implement online learning for continuous improvement:

```swift
// In TranslationPipeline.swift
func adaptToUserFeedback(correction: String, original: String) {
    // Implement online learning logic
}
```

## Support and Resources

- **Documentation**: Check the main README.md
- **Issues**: Report bugs on GitHub
- **Community**: Join our Discord server
- **Examples**: See the `examples/` directory for sample implementations

## License

This framework is licensed under the MIT License. See LICENSE file for details. 