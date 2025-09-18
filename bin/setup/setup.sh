#!/bin/bash

# Setup script for Document Processing with CrewAI and LangChain
# This script sets up the environment and dependencies
# Run this from the project root directory

set -e

# Ensure we're in the project root
if [[ ! -f "requirements.txt" || ! -d "src" ]]; then
    echo "❌ This script must be run from the project root directory"
    echo "Expected to find requirements.txt and src/ directory"
    exit 1
fi

echo "🚀 Setting up CrewAI-LangChain Document Processing Environment"
echo "============================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3.8+ and try again."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed."
    exit 1
fi

echo "✅ pip3 found"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Check if Tesseract is installed
echo "🔍 Checking for Tesseract OCR..."
if command -v tesseract &> /dev/null; then
    echo "✅ Tesseract found: $(tesseract --version | head -1)"
    TESSERACT_PATH=$(which tesseract)
    echo "   Path: $TESSERACT_PATH"
else
    echo "❌ Tesseract OCR not found"
    echo "📥 Installing Tesseract..."
    
    # Install Tesseract based on OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install tesseract
            TESSERACT_PATH=$(which tesseract)
        else
            echo "❌ Homebrew not found. Please install Homebrew first:"
            echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr
        TESSERACT_PATH=$(which tesseract)
    else
        echo "❌ Unsupported OS. Please install Tesseract manually."
        exit 1
    fi
fi

# Check if Ollama is installed
echo "🔍 Checking for Ollama..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama found: $(ollama --version)"
else
    echo "❌ Ollama not found"
    echo "📥 Installing Ollama..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            curl -fsSL https://ollama.ai/install.sh | sh
        fi
    else
        curl -fsSL https://ollama.ai/install.sh | sh
    fi
fi

# Update .env file with correct Tesseract path
echo "⚙️  Updating configuration..."
if [ -f ".env" ]; then
    # Update existing .env
    if grep -q "TESSERACT_CMD=" .env; then
        sed -i.bak "s|TESSERACT_CMD=.*|TESSERACT_CMD=$TESSERACT_PATH|" .env
    else
        echo "TESSERACT_CMD=$TESSERACT_PATH" >> .env
    fi
else
    # Create .env from template
    cp .env.example .env 2>/dev/null || true
    echo "TESSERACT_CMD=$TESSERACT_PATH" >> .env
fi

echo "✅ Configuration updated with Tesseract path: $TESSERACT_PATH"

# Start Ollama service
echo "🚀 Starting Ollama service..."
if pgrep -x "ollama" > /dev/null; then
    echo "✅ Ollama service already running"
else
    echo "🔄 Starting Ollama service in background..."
    ollama serve &
    sleep 3
fi

# Pull required model
echo "📥 Pulling Ollama model..."
if ollama list | grep -q "llama3.2-vision"; then
    echo "✅ llama3.2-vision model already available"
else
    echo "🔄 Downloading llama3.2-vision model (this may take a while)..."
    ollama pull llama3.2-vision
fi

# Test installation
echo "🧪 Testing installation..."
echo "Running comprehensive import test..."
python3 -c "
import sys
sys.path.append('src')
try:
    # Test core modules
    from main.models import DocumentType, ProcessingResult
    from main.utils import ImageProcessor
    print('✅ Core modules imported successfully')
    
    # Test progress collector system
    from main.progress_collector import ProgressCollectorMCP, ImplementationType
    print('✅ Progress collector system imported successfully')
    
    # Test essential packages
    import pandas
    import openpyxl
    print('✅ Essential packages (pandas, openpyxl) imported successfully')
    
    import pathlib
    import asyncio
    print('✅ Standard library modules imported successfully')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    print('💡 Try running: pip install -r requirements.txt')
    sys.exit(1)
"

# Test progress collector system specifically
echo "🧪 Testing Progress Collector System..."
python3 -c "
import sys
sys.path.append('.')
try:
    # Test imports first
    from src.main.progress_collector import ProgressCollectorMCP, ImplementationType, EventType
    print('✅ Progress collector imports working')
    
    # Test enhanced processors
    from src.main.enhanced_processors import EnhancedCrewAIProcessor, EnhancedLangChainProcessor, BatchProcessor
    print('✅ Enhanced processors imports working')
    
    # Create a test instance
    collector = ProgressCollectorMCP()
    print('✅ Progress collector instantiated successfully')
    
    # Test basic functionality with proper enum values
    proc_id = collector.start_processing(ImplementationType.CREWAI, 'test_doc.jpg', {'test': True})
    collector.log_info(ImplementationType.CREWAI, 'test_doc.jpg', 'Test message', 'testing')
    collector.end_processing(ImplementationType.CREWAI, 'test_doc.jpg', success=True)
    print('✅ Basic progress tracking working')
    
    # Test statistics generation
    stats = collector.get_summary_statistics()
    print(f'✅ Statistics working: {stats[\"session_info\"][\"total_events\"]} events')
    
    # Test Excel generation capability
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        try:
            report_path = collector.generate_excel_report(tmp.name)
            print('✅ Excel report generation working')
            os.unlink(tmp.name)  # Clean up
        except Exception as e:
            print(f'⚠️  Excel generation test failed: {e}')
    
    # Test JSON export capability  
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        try:
            json_path = collector.export_events_json(tmp.name)
            print('✅ JSON export working')
            os.unlink(tmp.name)  # Clean up
        except Exception as e:
            print(f'⚠️  JSON export test failed: {e}')
            
except Exception as e:
    print(f'❌ Progress collector test failed: {e}')
    print('💡 This may indicate missing dependencies or import issues')
    import traceback
    traceback.print_exc()
    sys.exit(1)
print('✅ Progress Collector System validation completed')
"

# Create sample directories and structure
echo "📁 Creating project structure..."
mkdir -p samples/input/passport
mkdir -p samples/input/idcard_classic  
mkdir -p samples/input/driverlicence
mkdir -p samples/output
mkdir -p bin/setup

# Make sure bin/run_all.sh is executable
if [ -f "bin/run_all.sh" ]; then
    chmod +x bin/run_all.sh
    echo "✅ Demo runner script made executable"
fi

# Make sure validation script is executable
if [ -f "bin/setup/validate_setup.sh" ]; then
    chmod +x bin/setup/validate_setup.sh
    echo "✅ Validation script made executable"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo "=============================================="
echo ""
echo "📋 Quick Start (Updated):"
echo "   1. Activate virtual environment: source venv/bin/activate"
echo "   2. Run comprehensive demo: ./bin/run_all.sh"
echo "   3. Run specific tests:"
echo "      • Simple test: ./bin/run_all.sh --simple"
echo "      • Mock demo: ./bin/run_all.sh --mock"
echo "      • Progress collector demo: ./bin/run_all.sh --progress"
echo "      • All 16 documents: ./bin/run_all.sh --comprehensive"
echo "      • Analysis only: ./bin/run_all.sh --analysis"
echo "      • Complete suite: ./bin/run_all.sh --full"
echo ""
echo "📊 Progress Collector System Features:"
echo "   • Event-driven architecture for tracking CrewAI vs LangChain"
echo "   • Real-time progress monitoring and metrics collection"
echo "   • Excel report generation with multiple sheets"
echo "   • JSON export for further analysis"
echo "   • Performance comparison and benchmarking"
echo "   • Batch processing support for multiple documents"
echo "   • Enhanced processor wrappers with automatic progress tracking"
echo ""
echo "📄 Sample documents structure:"
echo "   samples/input/passport/       - Passport documents"
echo "   samples/input/idcard_classic/ - ID card documents"
echo "   samples/input/driverlicence/  - Driver license documents"
echo "   samples/output/               - Generated reports and analysis"
echo ""
echo "🔧 Configuration and Tools:"
echo "   • Configuration file: .env"
echo "   • Demo runner help: ./bin/run_all.sh --help"
echo "   • Validation script: ./bin/setup/validate_setup.sh"
echo "   • Setup script location: ./bin/setup/setup.sh"
echo ""
echo "🆘 If you encounter issues:"
echo "   - Check README.md for troubleshooting"
echo "   - Ensure Ollama is running: ollama serve"
echo "   - Verify model is available: ollama list"
echo "   - Test imports: python -c 'from src.main.progress_collector import ProgressCollectorMCP'"
echo "   - Run validation: ./bin/setup/validate_setup.sh"
echo ""
echo "🎯 Recommended first run: ./bin/run_all.sh"
echo "   This runs comprehensive demo + analysis (16 documents)"
echo ""
echo "🚀 Production-Ready Features:"
echo "   • Event-driven progress tracking system"
echo "   • Comprehensive Excel reporting with charts and metrics"
echo "   • Batch processing and comparison analysis"
echo "   • Automated demo runner with multiple execution modes"
echo "   • Robust error handling and validation"
echo ""
echo "Happy document processing with comprehensive progress tracking! 🚀📊"
