# Document Processing with CrewAI and LangChain

This project implements document processing pipelines using both CrewAI and LangChain frameworks, integrated with local Ollama server and MCP (Model Context Protocol) for enhanced document classification, field extraction, accuracy assessment, and decision generation.

## Features

- **Document Classification**: Automatically classifies documents as ID cards, driver licenses, passports, or unknown
- **Field Extraction**: Extracts relevant fields based on document type
- **Accuracy Assessment**: Evaluates extraction quality and provides recommendations
- **Decision Generation**: Makes final accept/reject decisions based on quality metrics
- **MCP Integration**: Uses Model Context Protocol for enhanced processing
- **Dual Implementation**: Compare CrewAI vs LangChain approaches

## Architecture

### CrewAI Implementation
- **Collaborative Agents**: Multiple specialized agents work together
- **Agent Roles**: Classifier, Extractor, Accuracy Assessor, Decision Generator
- **Workflow**: Sequential task execution with agent collaboration

### LangChain Implementation  
- **Pipeline Approach**: Structured sequential processing
- **Tool-based**: Custom tools for each processing stage
- **Agent Executors**: Specialized agents with specific tool access

## Prerequisites

### System Requirements
- Python 3.8+
- macOS (tested) / Linux / Windows
- Tesseract OCR installed
- Ollama running locally

### Install Tesseract OCR

**macOS (Homebrew):**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### Install and Setup Ollama

1. **Install Ollama:**
   ```bash
   # macOS
   brew install ollama
   
   # Or download from: https://ollama.ai
   ```

2. **Start Ollama server:**
   ```bash
   ollama serve
   ```

3. **Pull required model:**
   ```bash
   ollama pull llama3.2-vision
   ```

## Installation

1. **Clone and setup:**
   ```bash
   cd /Users/moldovr/work/AI/ai-langchain
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Verify Tesseract path:**
   ```bash
   which tesseract
   # Update TESSERACT_CMD in .env if needed
   ```

## Usage

### Run CrewAI Demo
```bash
python src/main/crewai/demo.py
```

### Run LangChain Demo
```bash
python src/main/langchain/demo.py
```

### Run Comparison Demo
```bash
python src/full_demo.py
```

### Run Analyse Results
```bash
python  analyze_results.py
```

## Sample Documents

The system supports organized sample documents for testing and development:

### Directory Structure
```
samples/
├── input/
│   ├── driverlicence/
│   │   ├── driverlicence_front_specimen.png
│   │   ├── driverlicence_back_specimen.png
│   │   └── ... (other driver license samples)
│   ├── idcard_classic/
│   │   ├── idcard_classic_specimen.jpg
│   │   ├── idcard_classic_validation.png
│   │   └── ... (other ID card samples)
│   └── passport/
│       ├── passport_specimen.jpg
│       └── ... (other passport samples)
└── output/
    └── (processed results will be saved here)
```

### Adding Your Own Samples

1. **Place documents in appropriate folders:**
   - Driver licenses: `samples/input/driverlicence/`
   - ID cards: `samples/input/idcard_classic/`
   - Passports: `samples/input/passport/`

2. **Supported formats:**
   - JPG, JPEG, PNG
   - Recommended resolution: 1024x768 or higher
   - Clear, well-lit images work best

3. **Automatic detection:**
   - The demo scripts automatically find and process available samples
   - Run `python demo.py` for basic testing with real documents
   - Run `python src/full_demo.py` for comprehensive testing

### Batch Processing
```bash
# Process all samples with both frameworks
python src/full_demo.py

# Individual framework testing
python src/main/crewai/demo.py
python src/main/langchain/demo.py
```

### Process Custom Documents

```python
import asyncio
from src.main.crewai.document_processor import CrewAIDocumentProcessor
from src.main.langchain.document_processor import LangChainDocumentProcessor

async def process_custom_document():
    # Choose implementation
    processor = CrewAIDocumentProcessor()  # or LangChainDocumentProcessor()
    
    # Process document
    result = await processor.process_document("path/to/your/document.jpg")
    
    # Access results
    print(f"Document Type: {result.classification.document_type}")
    print(f"Confidence: {result.classification.confidence}")
    print(f"Decision: {'ACCEPTED' if result.final_decision.accept else 'REJECTED'}")

asyncio.run(process_custom_document())
```

## Document Types Supported

### ID Card Fields
- name, surname, date_of_birth, id_number
- address, nationality, gender
- issue_date, expiry_date

### Driver License Fields
- name, surname, date_of_birth, license_number
- address, class, issue_date, expiry_date, restrictions

### Passport Fields
- name, surname, passport_number, date_of_birth
- place_of_birth, nationality, gender
- issue_date, expiry_date

## Configuration

### Environment Variables (.env)
```bash
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2-vision

# MCP Configuration  
MCP_SERVER_URL=http://localhost:3000

# Document Processing
TESSERACT_CMD=/opt/homebrew/bin/tesseract
CONFIDENCE_THRESHOLD=0.85
```

### Ollama Models
The system uses `llama3.2-vision` for visual document processing. You can experiment with other models:

```bash
ollama pull llama3.2
ollama pull mistral
ollama pull codellama
```

## MCP Integration

The system integrates with Model Context Protocol for enhanced processing. Currently uses mock MCP responses - implement actual MCP server for production use.

Example MCP server setup:
```bash
# Start MCP server (implement your own)
node mcp-server.js --port 3000
```

## Performance Comparison

| Metric | CrewAI | LangChain |
|--------|--------|-----------|
| **Architecture** | Collaborative agents | Sequential pipeline |
| **Flexibility** | High (agent negotiation) | Medium (structured flow) |
| **Speed** | Moderate (collaboration overhead) | Fast (direct execution) |
| **Customization** | Agent personalities & roles | Tool composition |
| **Error Handling** | Agent consensus | Pipeline validation |

## Troubleshooting

### Common Issues

1. **Ollama connection error:**
   ```bash
   # Ensure Ollama is running
   ollama serve
   # Check if model is available
   ollama list
   ```

2. **Tesseract not found:**
   ```bash
   # Install Tesseract
   brew install tesseract
   # Update path in .env
   which tesseract
   ```

3. **Import errors:**
   ```bash
   # Ensure all dependencies are installed
   pip install -r requirements.txt
   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

4. **Low OCR accuracy:**
   - Use higher resolution images (300+ DPI)
   - Ensure good lighting and contrast
   - Avoid skewed or rotated documents
   - Clean/preprocess images before processing

### Performance Optimization

1. **Image preprocessing:**
   ```python
   # Implement custom preprocessing
   def preprocess_image(image):
       # Resize, denoise, sharpen
       return processed_image
   ```

2. **Batch processing:**
   ```python
   async def process_batch(documents):
       tasks = [processor.process_document(doc) for doc in documents]
       return await asyncio.gather(*tasks)
   ```

3. **Caching:**
   ```python
   # Cache OCR results to avoid reprocessing
   import functools
   
   @functools.lru_cache(maxsize=100)
   def cached_ocr(image_hash):
       return extract_text(image)
   ```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section
- Review Ollama documentation: https://ollama.ai/docs
- Check CrewAI docs: https://docs.crewai.com
- Check LangChain docs: https://python.langchain.com

## Acknowledgments

- Ollama team for local LLM serving
- CrewAI for collaborative agent framework  
- LangChain for agent orchestration tools
- Tesseract team for OCR capabilities
# ai-ram-crewai-langchain
# ai-ram-crewai-langchain
