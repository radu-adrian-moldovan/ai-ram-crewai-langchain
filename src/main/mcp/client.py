#!/usr/bin/env python3
"""
Simple MCP Client - Document Processing
Invokes the MCP server to process a document with a static image path
"""

import json
import requests
import logging
import os
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



# Load configuration

STATIC_IMAGE_PATH = "/Users/moldovr/work/AI/ai-ram-crewai-langchain/samples/input/idcard_classic/idcard_classic_JhonDoe.png"
MCP_SERVER_URL = "http://localhost:3000"

def send_mcp_request(method, params):
    """Send a request to the MCP server."""
    
    request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    request_payload = {
        "method": method,
        "params": params,
        "id": request_id
    }
    
    logger.info(f"Sending {method} request...")
    
    try:
        response = requests.post(
            MCP_SERVER_URL,
            json=request_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Request successful")
            return result
        else:
            logger.error(f"HTTP Error {response.status_code}")
            return {"error": f"HTTP {response.status_code}", "details": response.text}
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return {"error": str(e)}

def main():
    """Main function to process the static image."""
    
    print("=" * 60)
    print("🚀 Simple MCP Document Processing Client")
    print("=" * 60)
    print(f"📁 Image: {Path(STATIC_IMAGE_PATH).name}")
    print(f"🌐 Server: {MCP_SERVER_URL}")
    print()
    
    # Check if image file exists
    if not Path(STATIC_IMAGE_PATH).exists():
        print(f"❌ Error: Image file not found: {STATIC_IMAGE_PATH}")
        print("   Please update STATIC_IMAGE_PATH in the script")
        return False
    
    # 1. Health Check
    print("🏥 Health Check...")
    health_response = send_mcp_request("health/check", {})
    
    if "error" in health_response:
        print(f"❌ Server not responding: {health_response.get('error')}")
        print("   Make sure MCP server is running: ./bin/mcp_start.sh")
        return False
    
    print("✅ Server is healthy!")
    
    # 2. Document Classification
    print("\\n🔍 Document Classification...")
    classify_response = send_mcp_request("document/classify", {
        "image_path": STATIC_IMAGE_PATH
    })
    
    if "result" in classify_response:
        result = classify_response["result"]
        print(f"✅ Result : {result}")
        doc_type = result.get("document_type", "unknown")
        confidence = result.get("confidence", 0.0)
        print(f"✅ Document Type: {doc_type}")
        print(f"📊 Confidence: {confidence:.2f}")
    else:
        print(f"❌ Classification failed: {classify_response.get('error')}")
        doc_type = "unknown"
    
    # 3. Field Extraction
    print("\\n📋 Field Extraction...")
    extract_response = send_mcp_request("document/extract", {
        "image_path": STATIC_IMAGE_PATH,
        "document_type": doc_type
    })
    
    if "result" in extract_response:
        result = extract_response["result"]
        fields = result.get("extracted_fields", [])
        print(f"✅ Extracted {len(fields)} field(s):")
        
        for field in fields:
            field_name = field.get("field_name", "unknown")
            field_value = field.get("value", "")
            field_confidence = field.get("confidence", 0.0)
            
            # Display full values (truncation disabled)
            display_value = field_value
            # Original truncation code commented out for full display
            # display_value = (field_value[:50] + "...") if len(field_value) > 50 else field_value
            print(f"   • {field_name}: {display_value}")
            print(f"     Confidence: {field_confidence:.2f}")
    else:
        print(f"❌ Field extraction failed: {extract_response.get('error')}")
    
    # 4. Full Document Analysis
    print("\\n🔬 Full Document Analysis...")
    analyze_response = send_mcp_request("document/analyze", {
        "image_path": STATIC_IMAGE_PATH
    })
    
    if "result" in analyze_response:
        result = analyze_response["result"]
        
        # Extract key information
        extracted_text = result.get('extracted_text', '')
        ocr_confidence = result.get('ocr_confidence', 0.0)
        bounding_boxes = result.get('bounding_boxes', [])
        has_image = bool(result.get('image_base64'))
        
        print(f"✅ Analysis Results:")
        print(f"   📝 Text Length: {len(extracted_text)} characters")
        print(f"   📊 OCR Confidence: {ocr_confidence:.2f}")
        print(f"   📍 Text Regions: {len(bounding_boxes)}")
        print(f"   🖼️  Base64 Image: {'Yes' if has_image else 'No'}")
        
        # Show text preview
        if extracted_text:
            preview = (extracted_text[:100] + "...") if len(extracted_text) > 100 else extracted_text
            print(f"   📖 Text Preview: {preview}")
        else:
            print("   📖 Text Preview: No text detected")
            
    else:
        print(f"❌ Analysis failed: {analyze_response.get('error')}")
    
    print("\\n" + "=" * 60)
    print("✅ Processing completed!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\\n\\n⚠️  Interrupted by user.")
        exit(1)
    except Exception as e:
        logger.error(f"Client failed: {e}")
        print(f"❌ Client failed: {e}")
        exit(1)
