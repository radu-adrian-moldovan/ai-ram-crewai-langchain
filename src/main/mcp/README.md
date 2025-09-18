# MCP (Model Context Protocol) Server

This directory contains the MCP server implementation for document processing services.

## Files

- `__init__.py` - Package initialization
- `server.py` - MCP server implementation with CLI entry point
- `README.md` - This documentation

## Starting the Server

### Option 1: Using the startup script (recommended)
```bash
./bin/mcp_start.sh
```

### Option 2: Direct Python execution
```bash
python3 src/main/mcp/server.py
```

## Server Configuration

- **Host**: localhost
- **Port**: 3000
- **Protocol**: HTTP with JSON-RPC-like interface

## Available Methods

- `document/analyze` - Comprehensive document analysis
- `document/extract` - Extract fields from documents  
- `document/classify` - Classify document type
- `health/check` - Server health check
- `capabilities` - Get server capabilities

## Integration

The MCP server is integrated with the CrewAI pipeline through the `MCPTool` class in `src/main/crewai/document_processor.py`.

## Dependencies

- Python 3.8+
- opencv-python (for image processing)
- pytesseract (for OCR)
- All dependencies in requirements.txt
