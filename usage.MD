# Usage Guide - AI Document Processing System

## 🎯 Overview

This usage guide provides comprehensive instructions for running and using the AI Document Processing System. The system supports dual framework processing (CrewAI and LangChain) with Model Context Protocol (MCP) integration for enhanced document analysis.

## 🚀 Quick Start

### Basic Usage Commands

#### Run Default Comprehensive Demo
```bash
# Recommended: Run comprehensive demo + analysis
./bin/run_all.sh
```
**New Default Behavior**: Runs `--all` option (comprehensive + analysis) for optimal performance.

#### Individual Framework Demos
```bash
# CrewAI framework only
python src/main/crewai/demo.py

# LangChain framework only  
python src/main/langchain/demo.py

# Full comparison demo
python src/full_demo.py
```

### Complete Command Options
```bash
# All available run options
./bin/run_all.sh                # Default: comprehensive + analysis (--all)
./bin/run_all.sh --simple       # Basic functionality test only
./bin/run_all.sh --mock         # Mock demo with 3 sample documents
./bin/run_all.sh --comprehensive # All 16 documents processing only
./bin/run_all.sh --analysis     # Results analysis only
./bin/run_all.sh --all          # Comprehensive + analysis (same as default)
./bin/run_all.sh --full         # Complete suite (simple + mock + comprehensive + analysis)
./bin/run_all.sh --help         # Show usage information
```

## 📁 Document Processing

### Supported Document Types

The system supports three main document categories:

#### 1. ID Cards (`idcard_classic/`)
**Extracted Fields:**
- name, surname, date_of_birth, id_number
- address, nationality, gender
- issue_date, expiry_date

**Sample Files:**
- `idcard_classic_specimen.jpg`
- `idcard_classic_JhonDoe.png`
- `idcard_classic_Superman.jpg`
- `idcard_specimen_validation.png`

#### 2. Driver Licenses (`driverlicence/`)
**Extracted Fields:**
- name, surname, date_of_birth, license_number
- address, class, issue_date, expiry_date, restrictions

**Sample Files:**
- `driverlicence_front_specimen.png`
- `driverlicence_back_specimen.png`
- `driverlicence_front_ilie.jpg`
- `driverlicence_back_ilie.png`

#### 3. Passports (`passport/`)
**Extracted Fields:**
- name, surname, passport_number, date_of_birth
- place_of_birth, nationality, gender
- issue_date, expiry_date

**Sample Files:**
- `passport_specimen.jpg`
- `pasaport_aswad.jpg`
- `pasaport_Mihai.jpg`

### Sample Document Structure
```
samples/
├── input/
│   ├── driverlicence/           # Driver license documents
│   ├── idcard_classic/          # ID card documents  
│   └── passport/                # Passport documents
└── output/                      # Generated results
    ├── reports/                 # Excel and analysis reports
    ├── traces/                  # Individual processing traces
    └── comprehensive_*.xlsx     # Main comparison reports
```

### Adding Your Own Documents

#### 1. Organize Documents
Place your documents in the appropriate subdirectories:
```bash
# Driver licenses
cp your_license.jpg samples/input/driverlicence/

# ID cards
cp your_id.png samples/input/idcard_classic/

# Passports  
cp your_passport.jpg samples/input/passport/
```

#### 2. Supported Formats
- **File Types**: JPG, JPEG, PNG
- **Resolution**: 1024x768 or higher recommended
- **Quality**: Clear, well-lit images work best
- **Size Limit**: Configurable via `MAX_DOCUMENT_SIZE_MB` in .env

#### 3. Automatic Processing
```bash
# Process all documents in all subdirectories
./bin/run_all.sh --comprehensive

# Process with analysis
./bin/run_all.sh --all
```

## 🔧 Processing Modes

### 1. Simple Test Mode
**Purpose**: Quick functionality verification
```bash
./bin/run_all.sh --simple
```
- Tests basic system connectivity
- Verifies core components
- Quick validation (under 1 minute)

### 2. Mock Demo Mode  
**Purpose**: Test with sample data
```bash
./bin/run_all.sh --mock
```
- Processes 3 representative documents
- Fast execution for development/testing
- Includes both frameworks

### 3. Comprehensive Mode
**Purpose**: Full document processing
```bash
./bin/run_all.sh --comprehensive
```
- Processes ALL documents from all subdirectories
- Both CrewAI and LangChain frameworks
- Detailed comparison and metrics
- Complete trace generation

### 4. Analysis Only Mode
**Purpose**: Analyze existing results
```bash
./bin/run_all.sh --analysis
```
- Analyzes previously generated results
- Creates summary reports
- Generates performance metrics
- Useful for post-processing analysis

## 📊 Understanding Results

### Output Files Generated

#### 1. Excel Reports (`comprehensive_*.xlsx`)
**Location**: `samples/output/`

**Worksheets Include:**
- **Summary**: Overall system performance metrics
- **CrewAI Results**: Individual CrewAI processing results
- **LangChain Results**: Individual LangChain processing results  
- **Comparison**: Side-by-side framework comparison
- **Performance**: Processing time and accuracy metrics
- **Events Timeline**: Detailed processing timeline

#### 2. JSON Event Files (`comprehensive_events_*.json`)
**Location**: `samples/output/`

**Contains:**
- Raw processing data
- Event timeline with timestamps
- Framework-specific results
- Error logs and warnings
- System performance data

#### 3. Individual Trace Files
**Location**: `samples/output/traces/`

**Format**: `{framework}_{document}_{timestamp}.json`

**Examples:**
- `crewai_idcard_classic_JhonDoe_20250916_154024.json`
- `langchain_passport_specimen_20250916_154034.json`

**Content:**
- Step-by-step processing details
- Agent/chain execution traces
- Processing times per step
- Detailed results and confidence scores

### Reading Results

#### Excel Report Structure
```
📊 comprehensive_report_YYYYMMDD_HHMMSS.xlsx
├── 📝 Summary Sheet
│   ├── Total Documents Processed
│   ├── Success Rate by Framework
│   ├── Average Processing Time
│   └── Overall Accuracy Metrics
├── 🤖 CrewAI Results
│   ├── Document Path | Type | Confidence | Result | Time
│   └── Individual processing results
├── 🔗 LangChain Results  
│   ├── Document Path | Type | Confidence | Result | Time
│   └── Individual processing results
├── ⚖️ Comparison
│   ├── Side-by-side framework comparison
│   ├── Performance differences
│   └── Accuracy comparison
└── 📈 Performance Metrics
    ├── Processing time distribution
    ├── Success rate analysis
    └── Resource utilization
```

#### Key Metrics to Monitor
- **Processing Time**: Time taken per document
- **Confidence Score**: Classification/extraction confidence (0.0-1.0)
- **Success Rate**: Percentage of successful processings
- **Accuracy**: Quality of extracted fields
- **Framework Comparison**: CrewAI vs LangChain performance

## 🔍 Advanced Usage

### Custom Document Processing

#### Python API Usage
```python
import asyncio
from src.main.crewai.document_processor import CrewAIDocumentProcessor
from src.main.langchain.document_processor import LangChainDocumentProcessor

async def process_custom_document():
    # Choose framework
    processor = CrewAIDocumentProcessor()  # or LangChainDocumentProcessor()
    
    # Process single document
    result = await processor.process_document("path/to/your/document.jpg")
    
    # Access results
    print(f"Document Type: {result.classification.document_type}")
    print(f"Confidence: {result.classification.confidence}")
    print(f"Final Decision: {'ACCEPTED' if result.final_decision.accept else 'REJECTED'}")
    print(f"Processing Time: {result.processing_time:.2f}s")
    
    # Access extracted fields
    for field in result.extraction.extracted_fields:
        print(f"{field.field_name}: {field.value}")

# Run the processing
asyncio.run(process_custom_document())
```

#### Batch Processing
```python
import asyncio
from src.main.core.manager_processor import BatchProcessor
from src.main.core.progress_collector import ProgressCollectorMCP

async def batch_process_documents():
    # Initialize progress collector
    progress_collector = ProgressCollectorMCP()
    
    # Create batch processor
    batch_processor = BatchProcessor(progress_collector)
    
    # List of documents to process
    documents = [
        "samples/input/idcard_classic/idcard_classic_JhonDoe.png",
        "samples/input/passport/passport_specimen.jpg",
        "samples/input/driverlicence/driverlicence_front_specimen.png"
    ]
    
    # Process all documents
    results = await batch_processor.process_batch(documents)
    
    # Generate reports
    await batch_processor.generate_comparison_report(results)

asyncio.run(batch_process_documents())
```

### Framework Comparison Analysis

#### Understanding Framework Differences
```python
# Access comparison results
comparison_results = batch_processor.get_comparison_results()

for doc_path, comparison in comparison_results.items():
    crewai_result = comparison['crewai']
    langchain_result = comparison['langchain']
    
    print(f"\nDocument: {doc_path}")
    print(f"CrewAI - Time: {crewai_result.processing_time:.2f}s, Confidence: {crewai_result.classification.confidence:.2f}")
    print(f"LangChain - Time: {langchain_result.processing_time:.2f}s, Confidence: {langchain_result.classification.confidence:.2f}")
    
    # Performance comparison
    time_diff = abs(crewai_result.processing_time - langchain_result.processing_time)
    faster_framework = "CrewAI" if crewai_result.processing_time < langchain_result.processing_time else "LangChain"
    print(f"Faster: {faster_framework} (by {time_diff:.2f}s)")
```

## 🛠️ Troubleshooting Usage Issues

### Common Usage Problems

#### 1. Low OCR Accuracy
**Symptoms**: Poor field extraction, low confidence scores

**Solutions**:
```bash
# Use higher resolution images (300+ DPI)
# Ensure good lighting and contrast  
# Avoid skewed or rotated documents
# Clean/preprocess images before processing
```

**Image Preprocessing Tips**:
- Scan at minimum 300 DPI
- Use JPEG quality 95% or PNG
- Ensure document fills most of image frame
- Good lighting without shadows
- Straight document orientation

#### 2. Processing Failures
**Symptoms**: Error messages, incomplete results

**Check**:
```bash
# Verify Ollama is running
curl http://localhost:11434/api/tags

# Check MCP server
curl http://localhost:3000/health/check

# Review system logs
tail -f logs/system.log
```

#### 3. Performance Issues
**Symptoms**: Slow processing, timeouts

**Optimization**:
```bash
# Process smaller batches
./bin/run_all.sh --mock  # Test with fewer documents

# Check system resources
top | grep -E "(ollama|python)"

# Optimize image sizes (reduce to 1024x768 max)
# Close other applications to free memory
```

#### 4. Inconsistent Results
**Symptoms**: Different results between runs

**Causes & Solutions**:
- **Model variance**: Normal behavior for AI models
- **Image quality**: Ensure consistent input quality
- **System load**: Run during low system usage
- **Model temperature**: Check model configuration in .env

## 🔧 System Architecture & Recent Improvements

### Framework Integration
The system features:
- **Unified Tool System**: Framework-agnostic tool abstraction
- **MCP Service Layer**: Centralized Model Context Protocol integration
- **Dual Framework Support**: Both CrewAI and LangChain with consistent APIs

### Recent Fixes & Improvements

#### LangChain Template Fix
- **Issue**: Malformed template variables in prompts causing processing errors
- **Fix**: Simplified LangChain prompts with proper variable escaping
- **Result**: Improved reliability and reduced template variable mismatches

#### Manager Architecture Consolidation
- **Tool Management**: Consolidated tool adapters into unified `manager_tool.py`
- **MCP Integration**: Streamlined MCP service functions in `manager_mcp.py`
- **Configuration**: Centralized configuration management system

#### File Structure Updates
- **Demo File**: Updated from `comparison_demo.py` to `full_demo.py`
- **Import Updates**: All references updated across documentation
- **Command Consistency**: Unified command structure in all scripts

## 📈 Performance Metrics

### Framework Comparison Metrics

| Metric | CrewAI | LangChain | Notes |
|--------|--------|-----------|-------|
| **Architecture** | Collaborative agents | Sequential pipeline | CrewAI uses agent negotiation |
| **Flexibility** | High (agent roles) | Medium (structured flow) | CrewAI more customizable |
| **Speed** | Moderate | Fast | LangChain direct execution |
| **Customization** | Agent personalities | Tool composition | Different approaches |
| **Error Handling** | Agent consensus | Pipeline validation | Both robust |
| **Memory Usage** | Higher | Lower | CrewAI maintains agent state |
| **Scalability** | Good | Excellent | LangChain better for batch |

### Typical Performance Ranges
- **Processing Time**: 2-15 seconds per document
- **Classification Accuracy**: 85-95%
- **Field Extraction Success**: 80-90%
- **Overall Success Rate**: 90-95%

### Benchmark Results
```
Average Processing Times (16 documents):
├── CrewAI: 8.2s per document
├── LangChain: 5.7s per document
├── Total Batch: ~3-4 minutes
└── Analysis Generation: ~30 seconds
```

## 🎯 Best Practices

### 1. Document Preparation
- **Image Quality**: Use high-resolution, clear images
- **Lighting**: Ensure even lighting without shadows
- **Orientation**: Keep documents straight and properly oriented
- **Format**: Prefer PNG over JPEG for text documents
- **Size**: Balance quality vs processing speed

### 2. Processing Workflow
```bash
# Recommended processing sequence:
1. ./bin/run_all.sh --simple        # Verify system
2. ./bin/run_all.sh --mock          # Test with samples  
3. ./bin/run_all.sh --comprehensive # Full processing
4. ./bin/run_all.sh --analysis      # Generate reports
```

### 3. Result Analysis
- **Review Excel reports** for overall system performance
- **Check individual traces** for debugging specific documents
- **Compare frameworks** to understand performance characteristics
- **Monitor confidence scores** to identify problematic documents

### 4. System Maintenance
```bash
# Regular maintenance tasks:
# 1. Archive old results
mv samples/output/comprehensive_*.xlsx archive/

# 2. Clean trace files (optional)
find samples/output/traces -name "*.json" -mtime +7 -delete

# 3. Update models periodically
ollama pull llama3.2-vision:latest

# 4. Check disk space
du -sh samples/output/
```

## 🔄 Integration with Other Systems

### API Integration
The system can be integrated with external applications:

```python
# Example Flask API wrapper
from flask import Flask, request, jsonify
import asyncio

app = Flask(__name__)

@app.route('/process', methods=['POST'])
async def api_process_document():
    file = request.files['document']
    framework = request.form.get('framework', 'crewai')
    
    # Save uploaded file
    file_path = f"temp/{file.filename}"
    file.save(file_path)
    
    # Process document
    if framework == 'crewai':
        processor = CrewAIDocumentProcessor()
    else:
        processor = LangChainDocumentProcessor()
    
    result = await processor.process_document(file_path)
    
    # Return JSON response
    return jsonify({
        'document_type': result.classification.document_type.value,
        'confidence': result.classification.confidence,
        'extracted_fields': [
            {'name': field.field_name, 'value': field.value}
            for field in result.extraction.extracted_fields
        ],
        'processing_time': result.processing_time
    })
```

### Webhook Integration
```python
# Send results to external webhook
import httpx

async def send_webhook_notification(result, webhook_url):
    payload = {
        'event': 'document_processed',
        'document_type': result.classification.document_type.value,
        'success': result.final_decision.accept,
        'timestamp': datetime.now().isoformat()
    }
    
    async with httpx.AsyncClient() as client:
        await client.post(webhook_url, json=payload)
```

## 🔧 MCP Client Usage

### Direct MCP Processing

For direct document processing through the MCP server without the full framework:

1. **Start MCP Server**:
   ```bash
   ./bin/mcp_start.sh
   ```

2. **Configure Image Path**:
   Edit `src/main/mcp/config.env`:
   ```bash
   IMAGE_PATH=samples/input/idcard_classic/idcard_specimen_validation.png
   SERVER_URL=http://localhost:3000
   ```

3. **Run MCP Client**:
   ```bash
   python src/main/mcp/client.py
   ```

**Benefits:**
- Direct MCP server communication
- Lightweight processing
- Quick single document testing
- Simplified debugging

This usage guide provides comprehensive instructions for effectively using the AI Document Processing System. For setup and installation instructions, refer to the [Setup Guide](setup.MD).
