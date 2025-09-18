# Setup Guide - AI Document Processing System

## 🎯 Overview

This setup guide will help you install and configure the AI Document Processing System with CrewAI and LangChain frameworks, integrated with local Ollama server and MCP (Model Context Protocol).

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8+ (Python 3.11+ recommended)
- **Operating System**: macOS (tested) / Linux / Windows
- **Memory**: Minimum 8GB RAM (16GB+ recommended for optimal performance)
- **Storage**: At least 5GB free space for models and dependencies

### Required External Tools

#### 1. Tesseract OCR
Tesseract OCR is required for text extraction from document images.

**macOS (Homebrew):**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev
```

**Windows:**
- Download from: [Tesseract Windows Installer](https://github.com/UB-Mannheim/tesseract/wiki)
- Add Tesseract to your system PATH
- Note the installation directory for configuration

**Verify Installation:**
```bash
tesseract --version
which tesseract  # Note this path for .env configuration
```

#### 2. Ollama (Local LLM Server)
Ollama provides local language model serving for offline-first operation.

**Install Ollama:**
```bash
# macOS (Homebrew)
brew install ollama

# macOS/Linux (Direct download)
curl -fsSL https://ollama.ai/install.sh | sh

# Windows: Download from https://ollama.ai
```

**Start Ollama Server:**
```bash
# Start the server (keep running in background)
ollama serve
```

**Pull Required Models:**
```bash
# Vision model for document processing (recommended)
ollama pull llama3.2-vision

# Alternative text-only model
ollama pull llama3.1

# Verify models are installed
ollama list
```

## 🚀 Installation

### 1. Clone and Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd ai-ram-crewai-langchain

# Create and activate virtual environment
python -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
venv\\Scripts\\activate
```

### 2. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
pip list | grep -E "(crewai|langchain|ollama)"
```

### 3. Environment Configuration

#### Create Environment File
```bash
# Copy template
cp .env.example .env

# Edit configuration
nano .env  # or your preferred editor
```

#### Environment Variables (.env)
Configure the following variables in your `.env` file:

```bash
# ===== CORE SYSTEM CONFIGURATION =====
# Project root directory
PROJECT_ROOT=/Users/moldovr/work/AI/ai-ram-crewai-langchain

# ===== LLM CONFIGURATION (OFFLINE-FIRST) =====
# Primary: Local Ollama Server
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_VISION=llama3.2-vision:latest
OLLAMA_MODEL_TEXT=llama3.1:latest

# Fallback: OpenAI API (optional - set only if you have API key)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL_VISION=gpt-4-vision-preview
OPENAI_MODEL_TEXT=gpt-3.5-turbo

# ===== MCP SERVER CONFIGURATION =====
MCP_SERVER_URL=http://localhost:3000
MCP_SERVER_TIMEOUT=30

# ===== DOCUMENT PROCESSING =====
# Tesseract OCR path (update with your system path)
TESSERACT_CMD=/opt/homebrew/bin/tesseract  # macOS Homebrew
# TESSERACT_CMD=/usr/bin/tesseract         # Linux
# TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe  # Windows

# Processing settings
CONFIDENCE_THRESHOLD=0.85
MAX_DOCUMENT_SIZE_MB=10
SUPPORTED_FORMATS=jpg,jpeg,png,pdf

# ===== OUTPUT CONFIGURATION =====
OUTPUT_DIRECTORY=samples/output
ARCHIVE_OLD_REPORTS=true
EXCEL_REPORTS_ENABLED=true
JSON_EXPORT_ENABLED=true
TRACE_FILES_ENABLED=true

# ===== LOGGING =====
LOG_LEVEL=INFO
LOG_FILE=logs/system.log
```

### 4. Directory Structure Setup
```bash
# Create required directories
mkdir -p samples/input/{driverlicence,idcard_classic,passport}
mkdir -p samples/output/{traces,reports}
mkdir -p logs

# Set permissions (Unix systems)
chmod +x bin/*.sh
```

## 🔧 System Configuration

### 1. Centralized Configuration System
The system uses a centralized YAML configuration file for agent prompts and settings.

**Configuration File**: `src/main/core/prompts.yaml`

This file contains:
- **Agent Definitions**: Role, goal, backstory, and tools for each agent
- **Task Templates**: Description templates and expected outputs  
- **LangChain Prompts**: Human prompts specific to LangChain agents

### 2. Core Manager Architecture
The system includes centralized managers:
- **LLM Manager**: `src/main/core/manager_llm.py` - Centralized LLM access with offline-first priority
- **MCP Manager**: `src/main/core/manager_mcp.py` - Model Context Protocol integration
- **Tool Manager**: `src/main/core/manager_tool.py` - Unified tool system
- **Processor Manager**: `src/main/core/manager_processor.py` - Document processing coordination
- **Progress Manager**: `src/main/core/progress_collector.py` - Enhanced progress tracking
- **Report Manager**: `src/main/core/manager_report.py` - Report generation and export

### 3. MCP Server Setup
The system includes an integrated MCP server for enhanced document processing.

**Auto-start with demo** (recommended):
The MCP server starts automatically when running demos.

**Manual start**:
```bash
# Start MCP server
./bin/mcp_start.sh

# Or directly:
python src/main/mcp/server.py
```

**Server Configuration**:
- **Host**: localhost
- **Port**: 3000
- **Available Methods**:
  - `document/analyze` - Comprehensive document analysis
  - `document/extract` - Extract fields from documents
  - `document/classify` - Classify document type
  - `health/check` - Server health check

## 🔧 MCP Client Setup

The system includes a simple MCP client for direct document processing through the MCP server.

### MCP Client Configuration

1. **Configure the MCP Client**:
   Create or edit `src/main/mcp/config.env`:
   ```bash
   # Image path to process
   IMAGE_PATH=/path/to/your/image.png
   
   # MCP server URL (default: http://localhost:3000)
   SERVER_URL=http://localhost:3000
   ```

2. **Available Sample Images**:
   - **ID Cards**: `samples/input/idcard_classic/idcard_specimen_validation.png`
   - **Driver Licenses**: `samples/input/driverlicence/driverlicence_front_specimen.png`
   - **Passports**: `samples/input/passport/pasaport_specimen.jpg`

3. **Run MCP Client**:
   ```bash
   # Start MCP server first
   ./bin/mcp_start.sh
   
   # Run the client
   python src/main/mcp/client.py
   ```

## ✅ Verification

### 1. Quick System Check
```bash
# Run the verification script
python verify_offline.py
```

This script checks:
- ✅ Python environment and dependencies
- ✅ Ollama server connectivity
- ✅ Model availability
- ✅ Tesseract OCR installation
- ✅ Project structure and permissions
- ✅ Environment variables

### 2. Manual Verification Steps

#### Check Ollama Status
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Should return JSON with available models
```

#### Test Tesseract
```bash
# Test Tesseract installation
tesseract --version
echo "Test OCR" | tesseract stdin stdout
```

#### Verify Python Dependencies
```bash
# Test core imports
python -c "
import crewai
import langchain
import yaml
from src.main.core.manager_llm import get_llm_manager
print('✅ All core dependencies imported successfully')
"
```

### 3. Test Basic Functionality
```bash
# Test LLM manager
python test_llm_manager.py

# Run simple demo
./bin/run_all.sh --simple
```

## 🛠️ Troubleshooting

### Common Installation Issues

#### 1. Ollama Connection Errors
```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama if not running
ollama serve

# Check available models
ollama list

# Pull missing models
ollama pull llama3.2-vision
```

#### 2. Tesseract Not Found
```bash
# Install Tesseract (if missing)
# macOS:
brew install tesseract

# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# Update TESSERACT_CMD in .env with correct path
which tesseract
```

#### 3. Python Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\\Scripts\\activate  # Windows

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 4. Permission Errors (Unix Systems)
```bash
# Fix script permissions
chmod +x bin/*.sh

# Fix directory permissions
chmod -R 755 samples/
chmod -R 755 logs/
```

#### 5. Memory Issues
```bash
# For systems with limited RAM, use smaller models
ollama pull llama3.1  # Text-only model (smaller)

# Or adjust model settings in .env
OLLAMA_MODEL_VISION=llama3.1:latest
```

### Environment-Specific Issues

#### macOS
- Ensure Homebrew is installed and updated
- Use Homebrew paths for Tesseract: `/opt/homebrew/bin/tesseract`
- Check Xcode command line tools: `xcode-select --install`

#### Linux
- Install development packages: `sudo apt-get install build-essential`
- Use system paths for Tesseract: `/usr/bin/tesseract`
- Check Python development headers: `sudo apt-get install python3-dev`

#### Windows
- Use full paths in .env with escaped backslashes
- Ensure PowerShell execution policy allows scripts
- Consider using WSL for easier setup

## 🐛 Advanced Troubleshooting

### Common Issues and Solutions

#### LangChain Template Errors
**Problem**: `Input to ChatPromptTemplate is missing variables` errors
**Cause**: Malformed template variables in prompt files
**Solution**: The system has been updated with fixed prompt templates. If you encounter this:
1. Ensure you're using the latest `src/main/core/prompts.yaml`
2. Check for proper variable escaping in custom prompts
3. Verify template variables match input parameters

#### MCP Server Connection Issues
**Problem**: `Failed to connect to MCP server`
**Solution**:
1. Verify MCP server is running: `./bin/mcp_start.sh`
2. Check server logs: `tail -f logs/mcp_server.log`
3. Verify port 3000 is available: `lsof -i :3000`
4. Restart server if needed

#### Framework Import Errors
**Problem**: Module import failures for CrewAI or LangChain
**Cause**: Missing optional dependencies
**Solution**:
```bash
# Install specific framework
pip install crewai crewai-tools  # For CrewAI
pip install langchain langchain-community  # For LangChain

# Or install all optional dependencies
pip install -r requirements.txt
```

#### Tesseract Path Issues
**Problem**: `TesseractNotFoundError`
**Solution**:
1. Verify Tesseract installation: `tesseract --version`
2. Update .env with correct path:
   ```bash
   # Find tesseract path
   which tesseract
   
   # Update .env
   TESSERACT_CMD=/usr/local/bin/tesseract  # Your actual path
   ```

#### Performance Optimization
**Slow Processing**: 
- Ensure Ollama models are downloaded: `ollama pull llama3.2-vision`
- Increase RAM allocation for Ollama
- Use SSD storage for model files
- Close unnecessary applications

**High Memory Usage**:
- Process documents in smaller batches
- Use `--simple` mode for testing
- Monitor system resources: `htop` or Activity Monitor

### Development Notes

#### Architecture Updates
The system has undergone recent architectural improvements:
- **Manager Split**: Tool management separated into focused modules
- **MCP Integration**: Enhanced Model Context Protocol support
- **Configuration**: Centralized configuration management
- **Error Handling**: Improved error reporting and logging

For development and debugging, see individual component documentation in the `docs/` folder.

## 🔧 Advanced Configuration

### 1. Custom Model Configuration
```bash
# .env customization for different models
OLLAMA_MODEL_VISION=mistral:latest
OLLAMA_MODEL_TEXT=codellama:latest

# Pull custom models
ollama pull mistral
ollama pull codellama
```

### 2. Performance Tuning
```bash
# Increase processing limits
MAX_DOCUMENT_SIZE_MB=50
CONFIDENCE_THRESHOLD=0.90

# Enable/disable features
EXCEL_REPORTS_ENABLED=false
TRACE_FILES_ENABLED=false
```

### 3. Development Setup
```bash
# Enable debug logging
LOG_LEVEL=DEBUG

# Development-specific settings
ARCHIVE_OLD_REPORTS=false
```

## 🎯 Next Steps

After successful setup, proceed to the [Usage Guide](usage.MD) to learn how to:
- Run document processing demos
- Process custom documents
- Generate comparison reports
- Analyze system performance
- Use the batch processing features

]

For specific component issues:
- **Ollama**: https://ollama.ai/docs
- **CrewAI**: https://docs.crewai.com
- **LangChain**: https://python.langchain.com
- **Tesseract**: https://tesseract-ocr.github.io

The system is designed to be robust and self-diagnosing. Most issues can be resolved by following the verification steps and ensuring all prerequisites are correctly installed.
