#!/usr/bin/env python3
"""
Setup script for Real-Time Translation Framework

This script sets up the development environment and installs all necessary dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_python_dependencies():
    """Install Python dependencies"""
    return run_command("pip install -r requirements.txt", "Installing Python dependencies")

def setup_ios_environment():
    """Setup iOS development environment"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        print("🔄 Setting up iOS development environment...")
        
        # Check if Xcode is installed
        if not os.path.exists("/Applications/Xcode.app"):
            print("❌ Xcode is not installed. Please install Xcode from the App Store.")
            return False
        
        # Check if Xcode command line tools are installed
        if not run_command("xcode-select --print-path", "Checking Xcode command line tools"):
            print("🔄 Installing Xcode command line tools...")
            run_command("xcode-select --install", "Installing Xcode command line tools")
        
        print("✅ iOS development environment setup completed")
        return True
    else:
        print("⚠️  iOS development requires macOS with Xcode")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "evaluation_results",
        "training_data",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def setup_git_hooks():
    """Setup Git hooks for code quality"""
    hooks_dir = Path(".git/hooks")
    if hooks_dir.exists():
        # Create pre-commit hook
        pre_commit_hook = hooks_dir / "pre-commit"
        pre_commit_content = """#!/bin/sh
# Pre-commit hook for code quality
echo "Running code quality checks..."
python -m flake8 . --max-line-length=100 --exclude=venv,__pycache__,.git
python -m black --check .
"""
        pre_commit_hook.write_text(pre_commit_content)
        pre_commit_hook.chmod(0o755)
        print("✅ Git hooks configured")

def generate_sample_config():
    """Generate sample configuration files"""
    
    # Training config
    training_config = {
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
    
    import json
    with open("training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)
    
    print("✅ Generated sample configuration files")

def main():
    """Main setup function"""
    print("🚀 Setting up Real-Time Translation Framework")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("❌ Failed to install Python dependencies")
        sys.exit(1)
    
    # Setup iOS environment
    setup_ios_environment()
    
    # Setup Git hooks
    setup_git_hooks()
    
    # Generate sample config
    generate_sample_config()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Open RealTimeTranslation.xcodeproj in Xcode")
    print("2. Build and run the iOS app on your device")
    print("3. Train models using: python train_translator.py --source-lang es --target-lang en")
    print("4. Convert models to Core ML: python convert_to_coreml.py --model-type translation --model-path ./models/es_to_en_transformer.pth --output-path ./models/es_to_en_transformer.mlmodel")
    print("5. Evaluate models: python evaluate_model.py --model-type translation --model-path ./models/es_to_en_transformer.pth --tokenizer-path ./models/es_to_en_tokenizer")

if __name__ == "__main__":
    main() 