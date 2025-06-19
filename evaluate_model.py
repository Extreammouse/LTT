#!/usr/bin/env python3
"""
Model Evaluation Script

This script evaluates the performance of trained translation and speech recognition models.
"""

import os
import json
import argparse
import logging
import time
from typing import Dict, List, Tuple
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import coremltools as ct
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate model performance"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def evaluate_translation_model(self, model_path: str, tokenizer_path: str,
                                 test_data: List[Tuple[str, str]], 
                                 source_lang: str, target_lang: str) -> Dict:
        """Evaluate a translation model"""
        
        logger.info(f"Evaluating {source_lang} to {target_lang} translation model")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        
        results = {
            'source_language': source_lang,
            'target_language': target_lang,
            'total_samples': len(test_data),
            'translations': [],
            'metrics': {}
        }
        
        total_time = 0
        correct_translations = 0
        
        for i, (source_text, target_text) in enumerate(test_data):
            start_time = time.time()
            
            # Tokenize input
            inputs = tokenizer(source_text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=128, num_beams=4)
            
            # Decode output
            predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            
            # Calculate accuracy (simple exact match)
            is_correct = predicted_text.lower().strip() == target_text.lower().strip()
            if is_correct:
                correct_translations += 1
            
            results['translations'].append({
                'source': source_text,
                'target': target_text,
                'predicted': predicted_text,
                'correct': is_correct,
                'inference_time': inference_time
            })
            
            logger.info(f"Sample {i+1}/{len(test_data)}: {source_text} -> {predicted_text}")
        
        # Calculate metrics
        accuracy = correct_translations / len(test_data)
        avg_inference_time = total_time / len(test_data)
        
        results['metrics'] = {
            'accuracy': accuracy,
            'avg_inference_time': avg_inference_time,
            'total_inference_time': total_time
        }
        
        logger.info(f"Translation Accuracy: {accuracy:.4f}")
        logger.info(f"Average Inference Time: {avg_inference_time:.4f}s")
        
        return results
    
    def evaluate_speech_recognition_model(self, model_path: str, test_data: List[Tuple[np.ndarray, str]],
                                        language: str) -> Dict:
        """Evaluate a speech recognition model"""
        
        logger.info(f"Evaluating {language} speech recognition model")
        
        # Load model
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        
        results = {
            'language': language,
            'total_samples': len(test_data),
            'recognitions': [],
            'metrics': {}
        }
        
        total_time = 0
        correct_recognitions = 0
        
        for i, (audio_features, expected_text) in enumerate(test_data):
            start_time = time.time()
            
            # Prepare input
            inputs = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = model(inputs)
                predicted_text = self.decode_output(outputs)
            
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            
            # Calculate accuracy
            is_correct = predicted_text.lower().strip() == expected_text.lower().strip()
            if is_correct:
                correct_recognitions += 1
            
            results['recognitions'].append({
                'expected': expected_text,
                'predicted': predicted_text,
                'correct': is_correct,
                'inference_time': inference_time
            })
            
            logger.info(f"Sample {i+1}/{len(test_data)}: {expected_text} -> {predicted_text}")
        
        # Calculate metrics
        accuracy = correct_recognitions / len(test_data)
        avg_inference_time = total_time / len(test_data)
        
        results['metrics'] = {
            'accuracy': accuracy,
            'avg_inference_time': avg_inference_time,
            'total_inference_time': total_time
        }
        
        logger.info(f"Recognition Accuracy: {accuracy:.4f}")
        logger.info(f"Average Inference Time: {avg_inference_time:.4f}s")
        
        return results
    
    def decode_output(self, outputs):
        """Decode model outputs to text (simplified)"""
        # This is a simplified decoder
        # In practice, you'd use a proper decoder with vocabulary
        return "decoded text"  # Placeholder
    
    def evaluate_coreml_model(self, model_path: str, test_inputs: List[Dict], 
                            expected_outputs: List[str], model_type: str) -> Dict:
        """Evaluate a Core ML model"""
        
        logger.info(f"Evaluating Core ML {model_type} model")
        
        # Load model
        model = ct.models.MLModel(model_path)
        
        results = {
            'model_type': model_type,
            'total_samples': len(test_inputs),
            'predictions': [],
            'metrics': {}
        }
        
        total_time = 0
        correct_predictions = 0
        
        for i, (test_input, expected_output) in enumerate(zip(test_inputs, expected_outputs)):
            start_time = time.time()
            
            # Run prediction
            prediction = model.predict(test_input)
            
            end_time = time.time()
            inference_time = end_time - start_time
            total_time += inference_time
            
            # Process prediction based on model type
            if model_type == "translation":
                predicted_text = self.process_translation_prediction(prediction)
            else:
                predicted_text = self.process_recognition_prediction(prediction)
            
            # Calculate accuracy
            is_correct = predicted_text.lower().strip() == expected_output.lower().strip()
            if is_correct:
                correct_predictions += 1
            
            results['predictions'].append({
                'expected': expected_output,
                'predicted': predicted_text,
                'correct': is_correct,
                'inference_time': inference_time
            })
            
            logger.info(f"Sample {i+1}/{len(test_inputs)}: {expected_output} -> {predicted_text}")
        
        # Calculate metrics
        accuracy = correct_predictions / len(test_inputs)
        avg_inference_time = total_time / len(test_inputs)
        
        results['metrics'] = {
            'accuracy': accuracy,
            'avg_inference_time': avg_inference_time,
            'total_inference_time': total_time
        }
        
        logger.info(f"Core ML Model Accuracy: {accuracy:.4f}")
        logger.info(f"Average Inference Time: {avg_inference_time:.4f}s")
        
        return results
    
    def process_translation_prediction(self, prediction):
        """Process translation model prediction"""
        # Extract text from prediction
        if 'output' in prediction:
            return str(prediction['output'])
        return "translated text"  # Placeholder
    
    def process_recognition_prediction(self, prediction):
        """Process speech recognition model prediction"""
        # Extract text from prediction
        if 'output' in prediction:
            return str(prediction['output'])
        return "recognized text"  # Placeholder
    
    def generate_test_data(self, source_lang: str, target_lang: str, 
                          num_samples: int = 10) -> List[Tuple[str, str]]:
        """Generate test data for evaluation"""
        
        if f"{source_lang}-{target_lang}" == "es-en":
            test_data = [
                ("Hola, ¿cómo estás?", "Hello, how are you?"),
                ("Gracias por tu ayuda", "Thank you for your help"),
                ("¿Dónde está el baño?", "Where is the bathroom?"),
                ("Me gusta mucho este lugar", "I really like this place"),
                ("¿Cuál es tu nombre?", "What is your name?"),
                ("Buenos días", "Good morning"),
                ("Buenas noches", "Good night"),
                ("Por favor, ayúdame", "Please help me"),
                ("¿Hablas inglés?", "Do you speak English?"),
                ("No entiendo", "I don't understand")
            ]
        else:
            # Generic test data
            test_data = [
                ("Hello world", "Bonjour le monde"),
                ("How are you?", "Comment allez-vous?"),
                ("Thank you", "Merci"),
                ("Goodbye", "Au revoir"),
                ("Please help me", "S'il vous plaît aidez-moi")
            ]
        
        return test_data[:num_samples]
    
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to file"""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def plot_results(self, results: Dict, output_path: str):
        """Generate plots for evaluation results"""
        
        metrics = results['metrics']
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy plot
        ax1.bar(['Accuracy'], [metrics['accuracy']], color='skyblue')
        ax1.set_ylim(0, 1)
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        
        # Inference time plot
        ax2.bar(['Avg Inference Time'], [metrics['avg_inference_time']], color='lightcoral')
        ax2.set_title('Average Inference Time')
        ax2.set_ylabel('Time (seconds)')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Plots saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument("--model-type", type=str, required=True,
                       choices=["translation", "speech_recognition", "coreml"],
                       help="Type of model to evaluate")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the model")
    parser.add_argument("--tokenizer-path", type=str,
                       help="Path to the tokenizer (for translation models)")
    parser.add_argument("--source-lang", type=str, default="es",
                       help="Source language code")
    parser.add_argument("--target-lang", type=str, default="en",
                       help="Target language code")
    parser.add_argument("--language", type=str,
                       help="Language code (for speech recognition models)")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--num-samples", type=int, default=10,
                       help="Number of test samples")
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator()
    
    # Generate test data
    if args.model_type == "translation":
        test_data = evaluator.generate_test_data(
            args.source_lang, args.target_lang, args.num_samples
        )
        
        results = evaluator.evaluate_translation_model(
            args.model_path,
            args.tokenizer_path,
            test_data,
            args.source_lang,
            args.target_lang
        )
    
    elif args.model_type == "speech_recognition":
        # Generate dummy test data for speech recognition
        test_data = [(np.random.randn(13, 128), f"test text {i}") 
                     for i in range(args.num_samples)]
        
        results = evaluator.evaluate_speech_recognition_model(
            args.model_path,
            test_data,
            args.language
        )
    
    elif args.model_type == "coreml":
        # Generate test data for Core ML model
        test_inputs = [{"input": np.random.randn(1, 128)} for _ in range(args.num_samples)]
        expected_outputs = [f"expected output {i}" for i in range(args.num_samples)]
        
        results = evaluator.evaluate_coreml_model(
            args.model_path,
            test_inputs,
            expected_outputs,
            "translation"  # or "speech_recognition"
        )
    
    # Save results
    output_path = os.path.join(args.output_dir, f"{args.model_type}_evaluation.json")
    evaluator.save_results(results, output_path)
    
    # Generate plots
    plot_path = os.path.join(args.output_dir, f"{args.model_type}_evaluation.png")
    evaluator.plot_results(results, plot_path)

if __name__ == "__main__":
    main() 