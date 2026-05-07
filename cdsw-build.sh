#!/bin/bash
# CML Model Build Script
# This script runs during model build to install dependencies and train models

set -e  # Exit on any error

echo "============================================================"
echo "🚀 CML MODEL BUILD PROCESS STARTING"
echo "============================================================"

echo "🔧 Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

echo "📁 Environment check..."
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Available space: $(df -h . | tail -1 | awk '{print $4}')"

echo "📄 Repository contents:"
ls -la

echo "🧠 Training models in CML environment..."
echo "This will generate the fitted_preprocessor.joblib and all model files..."

# Run the training process to generate models
python main.py --generate-data --train-models

echo "📁 Checking generated model files..."
if [ -d "models" ]; then
    echo "✅ Models directory created successfully!"
    echo "📄 Model files generated:"
    ls -la models/
    
    # Verify critical files exist
    if [ -f "models/fitted_preprocessor.joblib" ]; then
        echo "✅ fitted_preprocessor.joblib generated successfully"
        echo "   Size: $(du -h models/fitted_preprocessor.joblib | cut -f1)"
    else
        echo "❌ CRITICAL: fitted_preprocessor.joblib not found!"
        exit 1
    fi
    
    if [ -f "models/random_forest_model.joblib" ]; then
        echo "✅ ML models generated successfully"
    else
        echo "❌ CRITICAL: ML model files not found!"
        exit 1
    fi
else
    echo "❌ CRITICAL: Models directory not created!"
    exit 1
fi

echo "📂 Ensuring utils directory is accessible..."
if [ -d "utils" ]; then
    echo "✅ Utils directory found"
    echo "📄 Utils contents:"
    ls -la utils/
else
    echo "❌ CRITICAL: Utils directory not found!"
    exit 1
fi

echo "============================================================"
echo "✅ CML BUILD COMPLETED SUCCESSFULLY!"
echo "📊 Build summary:"
echo "   - Dependencies installed"
echo "   - Models trained and saved"
echo "   - All required files present"
echo "============================================================"

echo "📄 Final directory contents:"
ls -la