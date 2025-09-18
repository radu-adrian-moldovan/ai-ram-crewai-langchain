#!/bin/bash

# Validation script for setup.sh
# Tests all the components that setup.sh is supposed to configure
# Run this from the project root directory

# Ensure we're in the project root
if [[ ! -f "requirements.txt" || ! -d "src" ]]; then
    echo "❌ This script must be run from the project root directory"
    echo "Expected to find requirements.txt and src/ directory"
    echo "Usage: ./bin/setup/validate_setup.sh"
    exit 1
fi

echo "🔍 Validating CrewAI-LangChain setup functionality..."
echo "=" * 50

# Test 1: Check virtual environment
echo "📦 Test 1: Virtual Environment"
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment active: $VIRTUAL_ENV"
else
    echo "❌ Virtual environment not active"
    echo "   Run: source venv/bin/activate"
fi

# Test 2: Check Python version
echo ""
echo "🐍 Test 2: Python Version"
python_version=$(python --version 2>&1)
echo "✅ Python: $python_version"

# Test 3: Check essential dependencies
echo ""
echo "📚 Test 3: Essential Dependencies"
python -c "
import pandas
import openpyxl
print('✅ pandas and openpyxl available')
"

# Test 4: Check progress collector system
echo ""
echo "📊 Test 4: Progress Collector System"
python -c "
import sys
sys.path.append('src')
from main.progress_collector import ProgressCollectorMCP, ImplementationType
collector = ProgressCollectorMCP()
print('✅ Progress collector system working')
"

# Test 5: Check project structure
echo ""
echo "📁 Test 5: Project Structure"
required_dirs=(
    "samples/input/passport"
    "samples/input/idcard_classic"
    "samples/input/driverlicence"
    "samples/output"
    "bin"
    "bin/setup"
    "src/main"
)

for dir in "${required_dirs[@]}"; do
    if [[ -d "$dir" ]]; then
        echo "✅ Directory exists: $dir"
    else
        echo "❌ Missing directory: $dir"
    fi
done

# Test 6: Check demo runner
echo ""
echo "🚀 Test 6: Demo Runner"
if [[ -f "bin/run_all.sh" && -x "bin/run_all.sh" ]]; then
    echo "✅ Demo runner exists and is executable"
    echo "   Test help: ./bin/run_all.sh --help"
else
    echo "❌ Demo runner missing or not executable"
fi

# Test 7: Check setup scripts
echo ""
echo "⚙️ Test 7: Setup Scripts"
if [[ -f "bin/setup/setup.sh" && -x "bin/setup/setup.sh" ]]; then
    echo "✅ Setup script exists and is executable: bin/setup/setup.sh"
else
    echo "❌ Setup script missing or not executable: bin/setup/setup.sh"
fi

if [[ -f "bin/setup/validate_setup.sh" && -x "bin/setup/validate_setup.sh" ]]; then
    echo "✅ Validation script exists and is executable: bin/setup/validate_setup.sh"
else
    echo "❌ Validation script missing or not executable: bin/setup/validate_setup.sh"
fi

# Test 8: Check configuration files
echo ""
echo "⚙️ Test 8: Configuration Files"
if [[ -f ".env" ]]; then
    echo "✅ .env file exists"
else
    echo "⚠️  .env file not found (may be created by setup.sh)"
fi

if [[ -f "requirements.txt" ]]; then
    echo "✅ requirements.txt exists"
else
    echo "❌ requirements.txt missing"
fi

# Test 9: Test Excel generation
echo ""
echo "📊 Test 9: Excel Generation Test"
python -c "
import sys
sys.path.append('.')
from src.main.progress_collector import ProgressCollectorMCP, ImplementationType
import tempfile
import os

collector = ProgressCollectorMCP()
# Add some test events
collector.start_processing(ImplementationType.CREWAI, 'test.jpg')
collector.end_processing(ImplementationType.CREWAI, 'test.jpg', success=True)

# Test Excel generation
with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
    try:
        collector.generate_excel_report(tmp.name)
        print('✅ Excel generation working')
        os.unlink(tmp.name)
    except Exception as e:
        print(f'❌ Excel generation failed: {e}')
"

# Test 10: Enhanced processors validation
echo ""
echo "🔧 Test 10: Enhanced Processors"
python -c "
import sys
sys.path.append('.')
try:
    from src.main.enhanced_processors import EnhancedCrewAIProcessor, EnhancedLangChainProcessor, BatchProcessor
    print('✅ Enhanced processors imports working')
except Exception as e:
    print(f'❌ Enhanced processors import failed: {e}')
"

echo ""
echo "🎉 Validation completed!"
echo ""
echo "📋 Summary:"
echo "   If all tests show ✅, setup.sh should work correctly"
echo "   If any tests show ❌, those components need attention"
echo ""
echo "🚀 To run setup: ./bin/setup/setup.sh"
echo "🎯 To run demos: ./bin/run_all.sh"
echo "📊 For help: ./bin/run_all.sh --help"
