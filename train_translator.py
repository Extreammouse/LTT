#!/usr/bin/env python3
"""
Real-Time Translation Model Training Script

This script trains transformer models for real-time speech translation.
It supports multiple language pairs and can generate Core ML models for iOS deployment.
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import TransformerEncoder, TransformerDecoder
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
import coremltools as ct
from datasets import load_dataset, Dataset as HFDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationDataset(Dataset):
    """Custom dataset for translation training"""
    
    def __init__(self, source_texts: List[str], target_texts: List[str], 
                 tokenizer, max_length: int = 128):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]
        
        # Tokenize source and target
        source_encoding = self.tokenizer(
            source_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze()
        }

class CustomTransformerModel(nn.Module):
    """Custom transformer model for translation"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8, 
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        tgt = self.embedding(tgt) * np.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        memory = self.transformer_encoder(src, src_mask)
        output = self.transformer_decoder(tgt, memory, tgt_mask)
        output = self.output_projection(output)
        
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TranslationTrainer:
    """Main trainer class for translation models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        
    def load_data(self, source_lang: str, target_lang: str) -> Tuple[List[str], List[str]]:
        """Load training data for the specified language pair"""
        
        # For demonstration, we'll use a simple dataset
        # In practice, you'd load from OPUS, WMT, or other sources
        
        if f"{source_lang}-{target_lang}" == "es-en":
            # Spanish to English sample data
            source_texts = [
                "Hola, ¿cómo estás?",
                "Gracias por tu ayuda",
                "¿Dónde está el baño?",
                "Me gusta mucho este lugar",
                "¿Cuál es tu nombre?",
                "Buenos días",
                "Buenas noches",
                "Por favor, ayúdame",
                "¿Hablas inglés?",
                "No entiendo"
            ]
            
            target_texts = [
                "Hello, how are you?",
                "Thank you for your help",
                "Where is the bathroom?",
                "I really like this place",
                "What is your name?",
                "Good morning",
                "Good night",
                "Please help me",
                "Do you speak English?",
                "I don't understand"
            ]
        
        elif f"{source_lang}-{target_lang}" == "en-es":
            # English to Spanish sample data
            source_texts = [
                "Hello, how are you?",
                "Thank you for your help",
                "Where is the bathroom?",
                "I really like this place",
                "What is your name?",
                "Good morning",
                "Good night",
                "Please help me",
                "Do you speak Spanish?",
                "I don't understand"
            ]
            
            target_texts = [
                "Hola, ¿cómo estás?",
                "Gracias por tu ayuda",
                "¿Dónde está el baño?",
                "Me gusta mucho este lugar",
                "¿Cuál es tu nombre?",
                "Buenos días",
                "Buenas noches",
                "Por favor, ayúdame",
                "¿Hablas español?",
                "No entiendo"
            ]
        
        else:
            # Generic dataset for other language pairs
            source_texts = [
                "Hello world",
                "How are you?",
                "Thank you",
                "Goodbye",
                "Please help me"
            ]
            
            target_texts = [
                "Bonjour le monde",
                "Comment allez-vous?",
                "Merci",
                "Au revoir",
                "S'il vous plaît aidez-moi"
            ]
        
        return source_texts, target_texts
    
    def setup_tokenizer(self, source_lang: str, target_lang: str):
        """Setup tokenizer for the language pair"""
        
        # Use a multilingual tokenizer
        model_name = "Helsinki-NLP/opus-mt-{}-{}".format(source_lang, target_lang)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            # Fallback to a general multilingual tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
        
        # Add special tokens if needed
        special_tokens = {
            'pad_token': '[PAD]',
            'unk_token': '[UNK]',
            'bos_token': '[BOS]',
            'eos_token': '[EOS]'
        }
        
        self.tokenizer.add_special_tokens(special_tokens)
    
    def create_model(self):
        """Create the transformer model"""
        
        vocab_size = self.tokenizer.vocab_size
        
        self.model = CustomTransformerModel(
            vocab_size=vocab_size,
            d_model=self.config.get('d_model', 512),
            nhead=self.config.get('nhead', 8),
            num_encoder_layers=self.config.get('num_encoder_layers', 6),
            num_decoder_layers=self.config.get('num_decoder_layers', 6),
            dim_feedforward=self.config.get('dim_feedforward', 2048),
            dropout=self.config.get('dropout', 0.1)
        )
        
        self.model.to(self.device)
    
    def train(self, source_lang: str, target_lang: str, output_dir: str):
        """Train the translation model"""
        
        logger.info(f"Training {source_lang} to {target_lang} translation model")
        
        # Load data
        source_texts, target_texts = self.load_data(source_lang, target_lang)
        
        # Setup tokenizer
        self.setup_tokenizer(source_lang, target_lang)
        
        # Create model
        self.create_model()
        
        # Create dataset
        dataset = TranslationDataset(
            source_texts=source_texts,
            target_texts=target_texts,
            tokenizer=self.tokenizer,
            max_length=self.config.get('max_length', 128)
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 4),
            shuffle=True
        )
        
        # Setup training
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4)
        )
        
        # Training loop
        num_epochs = self.config.get('num_epochs', 10)
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(input_ids, labels)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save model
        self.save_model(output_dir, source_lang, target_lang)
        
        logger.info(f"Training completed. Model saved to {output_dir}")
    
    def save_model(self, output_dir: str, source_lang: str, target_lang: str):
        """Save the trained model"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save PyTorch model
        model_path = os.path.join(output_dir, f"{source_lang}_to_{target_lang}_transformer.pth")
        torch.save(self.model.state_dict(), model_path)
        
        # Save tokenizer
        tokenizer_path = os.path.join(output_dir, f"{source_lang}_to_{target_lang}_tokenizer")
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # Save model config
        config_path = os.path.join(output_dir, f"{source_lang}_to_{target_lang}_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def convert_to_coreml(self, output_dir: str, source_lang: str, target_lang: str):
        """Convert the trained model to Core ML format"""
        
        logger.info(f"Converting {source_lang} to {target_lang} model to Core ML")
        
        # Load the trained model
        model_path = os.path.join(output_dir, f"{source_lang}_to_{target_lang}_transformer.pth")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Create example input
        example_input = torch.randint(0, self.tokenizer.vocab_size, (1, 128))
        
        # Trace the model
        traced_model = torch.jit.trace(self.model, example_input)
        
        # Convert to Core ML
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(name="input", shape=example_input.shape)],
            minimum_deployment_target=ct.target.iOS15
        )
        
        # Save Core ML model
        coreml_path = os.path.join(output_dir, f"{source_lang}_to_{target_lang}_transformer.mlmodel")
        coreml_model.save(coreml_path)
        
        logger.info(f"Core ML model saved to {coreml_path}")

def main():
    parser = argparse.ArgumentParser(description="Train translation models")
    parser.add_argument("--source-lang", type=str, default="es", help="Source language code")
    parser.add_argument("--target-lang", type=str, default="en", help="Target language code")
    parser.add_argument("--output-dir", type=str, default="./models", help="Output directory")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--convert-coreml", action="store_true", help="Convert to Core ML")
    
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'd_model': 512,
        'nhead': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'max_length': 128,
        'batch_size': 4,
        'learning_rate': 1e-4,
        'num_epochs': 10
    }
    
    # Load custom config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Create trainer and train model
    trainer = TranslationTrainer(config)
    trainer.train(args.source_lang, args.target_lang, args.output_dir)
    
    # Convert to Core ML if requested
    if args.convert_coreml:
        trainer.convert_to_coreml(args.output_dir, args.source_lang, args.target_lang)

if __name__ == "__main__":
    main() 