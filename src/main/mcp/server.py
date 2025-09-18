#!/usr/bin/env python3
"""
Minimal MCP (Model Context Protocol) Server Implementation
Provides basic document processing services on localhost:3000
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
import sys
import base64
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from main.core.utils import ImageProcessor

class MCPServer:
    """Minimal MCP Server for document processing services."""
    
    def __init__(self, host: str = "localhost", port: int = 3000):
        self.host = host
        self.port = port
        self.image_processor = ImageProcessor()
        self.server = None
        
    async def handle_request(self, reader, writer):
        """Handle incoming MCP requests."""
        try:
            # Read the request
            data = await reader.read(8192)  # 8KB buffer - standard size for network operations
            if not data:
                return
                
            request_text = data.decode('utf-8')
            logger.info(f"Received request: {request_text[:100]}...")
            
            # Parse JSON request
            try:
                request = json.loads(request_text)
            except json.JSONDecodeError:
                # Try to extract JSON from HTTP request
                if '\r\n\r\n' in request_text:
                    body = request_text.split('\r\n\r\n', 1)[1]
                    request = json.loads(body)
                else:
                    raise
            
            # Process the request
            response = await self.process_request(request)
            
            # Send HTTP response
            response_json = json.dumps(response, indent=2)
            http_response = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: application/json\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n"
                "Access-Control-Allow-Headers: Content-Type\r\n"
                f"Content-Length: {len(response_json)}\r\n"
                "\r\n"
                f"{response_json}"
            )
            
            writer.write(http_response.encode('utf-8'))
            await writer.drain()
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            error_response = {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            error_json = json.dumps(error_response)
            http_error = (
                "HTTP/1.1 500 Internal Server Error\r\n"
                "Content-Type: application/json\r\n"
                f"Content-Length: {len(error_json)}\r\n"
                "\r\n"
                f"{error_json}"
            )
            writer.write(http_error.encode('utf-8'))
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process MCP request and return response."""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id", "unknown")
        
        logger.info(f"Processing method: {method} with ID: {request_id}")
        
        # Route to appropriate handler
        if method == "document/analyze":
            return await self.analyze_document(params, request_id)
        elif method == "document/extract":
            return await self.extract_fields(params, request_id)
        elif method == "document/classify":
            return await self.classify_document(params, request_id)
        elif method == "health/check":
            return await self.health_check(params, request_id)
        elif method == "capabilities":
            return await self.get_capabilities(params, request_id)
        else:
            return {
                "id": request_id,
                "error": f"Unknown method: {method}",
                "available_methods": [
                    "document/analyze",
                    "document/extract", 
                    "document/classify",
                    "health/check",
                    "capabilities"
                ]
            }
    
    async def analyze_document(self, params: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Analyze document image and return comprehensive results."""
        try:
            image_path = params.get("image_path")
            if not image_path:
                return {
                    "id": request_id,
                    "error": "image_path parameter required"
                }
            
            # Load and process image
            image = self.image_processor.load_image(image_path)
            if image is None:
                return {
                    "id": request_id,
                    "error": f"Could not load image: {image_path}"
                }
            
            # Extract text using OCR
            text_data = self.image_processor.extract_text_with_ocr(image)
            base64_image = self.image_processor.image_to_base64(image)
            
            # Basic document type detection based on text content
            extracted_text = " ".join(text_data.get('text', [])).lower()
            document_type = self._detect_document_type(extracted_text)
            
            return {
                "id": request_id,
                "result": {
                    "document_type": document_type,
                    "confidence": 0.85,
                    "extracted_text": " ".join(text_data.get('text', [])),
                    "ocr_confidence": sum(text_data.get('confidence', [])) / max(len(text_data.get('confidence', [])), 1),
                    "bounding_boxes": text_data.get('bounding_boxes', []),
                    "image_base64": base64_image,
                    "processing_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing document: {e}")
            return {
                "id": request_id,
                "error": f"Document analysis failed: {str(e)}"
            }
    
    async def extract_fields(self, params: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Extract specific fields from document."""
        try:
            image_path = params.get("image_path")
            document_type = params.get("document_type", "unknown")
            
            if not image_path:
                return {
                    "id": request_id,
                    "error": "image_path parameter required"
                }
            
            # Load and process image
            image = self.image_processor.load_image(image_path)
            if image is None:
                return {
                    "id": request_id,
                    "error": f"Could not load image: {image_path}"
                }
            
            # Extract text
            text_data = self.image_processor.extract_text_with_ocr(image)
            extracted_text = " ".join(text_data.get('text', []))
            
            # Extract fields based on document type
            fields = self._extract_fields_by_type(extracted_text, document_type)
            
            return {
                "id": request_id,
                "result": {
                    "document_type": document_type,
                    "extracted_fields": fields,
                    "total_fields_found": len(fields),
                    "processing_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting fields: {e}")
            return {
                "id": request_id,
                "error": f"Field extraction failed: {str(e)}"
            }
    
    async def classify_document(self, params: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Classify document type."""
        try:
            image_path = params.get("image_path")
            
            if not image_path:
                return {
                    "id": request_id,
                    "error": "image_path parameter required"
                }
            
            # Load and process image
            image = self.image_processor.load_image(image_path)
            if image is None:
                return {
                    "id": request_id,
                    "error": f"Could not load image: {image_path}"
                }
            
            # Extract text for classification
            text_data = self.image_processor.extract_text_with_ocr(image)
            extracted_text = " ".join(text_data.get('text', [])).lower()
            
            # Classify document
            document_type = self._detect_document_type(extracted_text)
            confidence = self._calculate_confidence(extracted_text, document_type)
            
            return {
                "id": request_id,
                "result": {
                    "document_type": document_type,
                    "confidence": confidence,
                    "reasoning": f"Detected based on text content analysis",
                    "processing_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error classifying document: {e}")
            return {
                "id": request_id,
                "error": f"Document classification failed: {str(e)}"
            }
    
    async def health_check(self, params: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "id": request_id,
            "result": {
                "status": "healthy",
                "server": "MCP Document Processing Server",
                "version": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "capabilities": [
                    "document/analyze",
                    "document/extract",
                    "document/classify"
                ]
            }
        }
    
    async def get_capabilities(self, params: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """Get server capabilities."""
        return {
            "id": request_id,
            "result": {
                "methods": [
                    {
                        "name": "document/analyze",
                        "description": "Comprehensive document analysis including OCR and classification",
                        "parameters": ["image_path"]
                    },
                    {
                        "name": "document/extract", 
                        "description": "Extract specific fields from documents",
                        "parameters": ["image_path", "document_type"]
                    },
                    {
                        "name": "document/classify",
                        "description": "Classify document type",
                        "parameters": ["image_path"]
                    },
                    {
                        "name": "health/check",
                        "description": "Server health check",
                        "parameters": []
                    }
                ],
                "supported_document_types": ["id_card", "driver_license", "passport", "unknown"],
                "ocr_enabled": True,
                "image_formats": ["jpg", "jpeg", "png", "bmp", "tiff"]
            }
        }
    
    def _detect_document_type(self, text: str) -> str:
        """Basic document type detection based on text content."""
        text_lower = text.lower()
        
        # Check for passport indicators
        if any(word in text_lower for word in ['passport', 'pasaport', 'república', 'republic']):
            return "passport"
        
        # Check for driver license indicators  
        if any(word in text_lower for word in ['driver', 'license', 'driving', 'permis', 'conducere']):
            return "driver_license"
        
        # Check for ID card indicators
        if any(word in text_lower for word in ['identity', 'identitate', 'carte', 'buletin', 'id card']):
            return "id_card"
        
        return "unknown"
    
    def _calculate_confidence(self, text: str, document_type: str) -> float:
        """Calculate confidence score for document classification."""
        text_lower = text.lower()
        
        confidence_keywords = {
            "passport": ['passport', 'pasaport', 'república', 'republic', 'nationality'],
            "driver_license": ['driver', 'license', 'driving', 'permis', 'conducere', 'class'],
            "id_card": ['identity', 'identitate', 'carte', 'buletin', 'id card', 'cnp']
        }
        
        if document_type in confidence_keywords:
            matches = sum(1 for keyword in confidence_keywords[document_type] if keyword in text_lower)
            return min(0.6 + (matches * 0.1), 0.95)
        
        return 0.3
    
    def _extract_fields_by_type(self, text: str, document_type: str) -> list:
        """Extract fields based on document type."""
        fields = []
        
        # This is a simplified field extraction - in a real implementation,
        # you would use more sophisticated pattern matching and NLP
        
        words = text.split()
        
        # Common field extraction patterns - full text extraction without truncation
        if document_type == "id_card":
            fields.extend([
                {"field_name": "extracted_text", "value": text, "confidence": 0.8},
                {"field_name": "document_type", "value": "id_card", "confidence": 0.9}
            ])
        elif document_type == "driver_license":
            fields.extend([
                {"field_name": "extracted_text", "value": text, "confidence": 0.8},
                {"field_name": "document_type", "value": "driver_license", "confidence": 0.9}
            ])
        elif document_type == "passport":
            fields.extend([
                {"field_name": "extracted_text", "value": text, "confidence": 0.8},
                {"field_name": "document_type", "value": "passport", "confidence": 0.9}
            ])
        else:
            fields.append({
                "field_name": "extracted_text", 
                "value": text, 
                "confidence": 0.5
            })
        
        return fields
    
    async def start_server(self):
        """Start the MCP server."""
        logger.info(f"Starting MCP server on {self.host}:{self.port}")
        
        self.server = await asyncio.start_server(
            self.handle_request,
            self.host,
            self.port
        )
        
        addr = self.server.sockets[0].getsockname()
        logger.info(f"MCP server running on {addr[0]}:{addr[1]}")
        
        async with self.server:
            await self.server.serve_forever()
    
    async def stop_server(self):
        """Stop the MCP server."""
        if self.server:
            logger.info("Stopping MCP server...")
            self.server.close()
            await self.server.wait_closed()
            logger.info("MCP server stopped")

async def main():
    """Main function to run the MCP server."""
    server = MCPServer()
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
        await server.stop_server()
    except Exception as e:
        logger.error(f"Server error: {e}")
        await server.stop_server()

if __name__ == "__main__":
    asyncio.run(main())
