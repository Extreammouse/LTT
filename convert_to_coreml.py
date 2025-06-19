#!/usr/bin/env python3
"""
Core ML Conversion Script

This script converts trained PyTorch models to Core ML format for iOS deployment.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import coremltools as ct
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelConverter:
    """Convert PyTorch models to Core ML format"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def convert_translation_model(self, model_path: str, tokenizer_path: str, 
                                output_path: str, source_lang: str, target_lang: str):
        """Convert a translation model to Core ML format"""
        
        logger.info(f"Converting {source_lang} to {target_lang} translation model")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load model
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.eval()
        
        # Create example inputs
        example_text = "Hello world"
        inputs = tokenizer(example_text, return_tensors="pt", padding=True, truncation=True)
        
        # Trace the model
        traced_model = torch.jit.trace(model, (inputs['input_ids'], inputs['attention_mask']))
        
        # Convert to Core ML
        coreml_model = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="input_ids", shape=inputs['input_ids'].shape),
                ct.TensorType(name="attention_mask", shape=inputs['attention_mask'].shape)
            ],
            minimum_deployment_target=ct.target.iOS15,
            compute_units=ct.ComputeUnit.CPU_AND_NE
        )
        
        # Add metadata
        coreml_model.author = "Real-Time Translation Framework"
        coreml_model.license = "MIT"
        coreml_model.short_description = f"Translation model for {source_lang} to {target_lang}"
        coreml_model.version = "1.0"
        
        # Save Core ML model
        coreml_model.save(output_path)
        
        logger.info(f"Core ML model saved to {output_path}")
    
    def convert_speech_recognition_model(self, model_path: str, output_path: str, 
                                       language: str):
        """Convert a speech recognition model to Core ML format"""
        
        logger.info(f"Converting {language} speech recognition model")
        
        # Load model (assuming it's a PyTorch model)
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        
        # Create example input (audio features)
        example_input = torch.randn(1, 13, 128)  # MFCC features
        
        # Trace the model
        traced_model = torch.jit.trace(model, example_input)
        
        # Convert to Core ML
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="audio_features", shape=example_input.shape)],
            minimum_deployment_target=ct.target.iOS15,
            compute_units=ct.ComputeUnit.CPU_AND_NE
        )
        
        # Add metadata
        coreml_model.author = "Real-Time Translation Framework"
        coreml_model.license = "MIT"
        coreml_model.short_description = f"Speech recognition model for {language}"
        coreml_model.version = "1.0"
        
        # Save Core ML model
        coreml_model.save(output_path)
        
        logger.info(f"Core ML model saved to {output_path}")
    
    def optimize_model(self, model_path: str, output_path: str):
        """Optimize a Core ML model for better performance"""
        
        logger.info(f"Optimizing model: {model_path}")
        
        # Load the model
        model = ct.models.MLModel(model_path)
        
        # Optimize for performance
        optimized_model = ct.compression_utils.affine_quantize_weights(
            model, mode="linear"
        )
        
        # Save optimized model
        optimized_model.save(output_path)
        
        logger.info(f"Optimized model saved to {output_path}")
    
    def validate_model(self, model_path: str, test_inputs: Dict[str, Any]):
        """Validate a Core ML model with test inputs"""
        
        logger.info(f"Validating model: {model_path}")
        
        # Load the model
        model = ct.models.MLModel(model_path)
        
        # Run predictions
        predictions = model.predict(test_inputs)
        
        logger.info("Model validation completed successfully")
        return predictions

def main():
    parser = argparse.ArgumentParser(description="Convert models to Core ML format")
    parser.add_argument("--model-type", type=str, required=True, 
                       choices=["translation", "speech_recognition"],
                       help="Type of model to convert")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the PyTorch model")
    parser.add_argument("--tokenizer-path", type=str,
                       help="Path to the tokenizer (for translation models)")
    parser.add_argument("--output-path", type=str, required=True,
                       help="Output path for Core ML model")
    parser.add_argument("--source-lang", type=str, default="es",
                       help="Source language code")
    parser.add_argument("--target-lang", type=str, default="en",
                       help="Target language code")
    parser.add_argument("--language", type=str,
                       help="Language code (for speech recognition models)")
    parser.add_argument("--optimize", action="store_true",
                       help="Optimize the model for better performance")
    parser.add_argument("--validate", action="store_true",
                       help="Validate the converted model")
    
    args = parser.parse_args()
    
    converter = ModelConverter()
    
    if args.model_type == "translation":
        if not args.tokenizer_path:
            raise ValueError("Tokenizer path is required for translation models")
        
        converter.convert_translation_model(
            args.model_path,
            args.tokenizer_path,
            args.output_path,
            args.source_lang,
            args.target_lang
        )
    
    elif args.model_type == "speech_recognition":
        if not args.language:
            raise ValueError("Language is required for speech recognition models")
        
        converter.convert_speech_recognition_model(
            args.model_path,
            args.output_path,
            args.language
        )
    
    # Optimize if requested
    if args.optimize:
        optimized_path = args.output_path.replace(".mlmodel", "_optimized.mlmodel")
        converter.optimize_model(args.output_path, optimized_path)
    
    # Validate if requested
    if args.validate:
        # Create test inputs based on model type
        if args.model_type == "translation":
            test_inputs = {
                "input_ids": [[1, 2, 3, 4, 5]],
                "attention_mask": [[1, 1, 1, 1, 1]]
            }
        else:
            test_inputs = {
                "audio_features": [[[0.1] * 128] * 13]
            }
        
        converter.validate_model(args.output_path, test_inputs)

if __name__ == "__main__":
    main() 