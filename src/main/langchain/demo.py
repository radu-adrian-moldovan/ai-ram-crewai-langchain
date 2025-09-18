#!/usr/bin/env python3
"""
LangChain Document Processing Demo

This script demonstrates the LangChain implementation for document processing
using local Ollama server and MCP integration.
"""

import asyncio
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.main.langchain.document_processor import LangChainDocumentProcessor
from src.main.core.models import ProcessingResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def demo_langchain_processing():
    """Demonstrate LangChain document processing."""
    
    print("=" * 60)
    print("LangChain Document Processing Demo")
    print("=" * 60)
    
    try:
        # Initialize processor
        print("Initializing LangChain Document Processor...")
        processor = LangChainDocumentProcessor(
            ollama_url="http://localhost:11434",
            mcp_url="http://localhost:3000"
        )
        
        # Sample document path (replace with actual image)
        sample_documents = [
            "sample_id_card.jpg",
            "sample_driver_license.jpg", 
            "sample_passport.jpg"
        ]
        
        for doc_path in sample_documents:
            if Path(doc_path).exists():
                print(f"\nProcessing document: {doc_path}")
                print("-" * 40)
                
                start_time = datetime.now()
                
                try:
                    result = await processor.process_document(doc_path)
                    
                    end_time = datetime.now()
                    processing_time = (end_time - start_time).total_seconds()
                    
                    # Display results
                    print_processing_results(result, processing_time)
                    
                except Exception as e:
                    logger.error(f"Error processing {doc_path}: {e}")
            else:
                print(f"Sample document {doc_path} not found. Creating mock result...")
                await demo_mock_processing()
    
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print("Make sure Ollama is running on http://localhost:11434")
        print("To start Ollama: ollama serve")

async def demo_mock_processing():
    """Demo with mock data when no sample images are available."""
    print("Running mock processing demonstration...")
    
    # This would normally process a real image
    mock_result = await create_mock_result()
    print_processing_results(mock_result, 3.2)

async def create_mock_result() -> ProcessingResult:
    """Create a mock processing result for demonstration."""
    from src.main.core.models import (
        DocumentClassification, DocumentExtraction, ExtractedField,
        AccuracyAssessment, FinalDecision, DocumentType
    )
    
    classification = DocumentClassification(
        document_type=DocumentType.DRIVER_LICENSE,
        confidence=0.88,
        reasoning="Document contains 'DRIVER LICENSE' header, motor vehicle authority logo, and standard license field layout"
    )
    
    extraction = DocumentExtraction(
        extracted_fields=[
            ExtractedField(field_name="name", value="Jane Marie Smith", confidence=0.93),
            ExtractedField(field_name="surname", value="Smith", confidence=0.96),
            ExtractedField(field_name="license_number", value="DL987654321", confidence=0.91),
            ExtractedField(field_name="date_of_birth", value="1990-07-22", confidence=0.87),
            ExtractedField(field_name="address", value="123 Main St, Anytown, ST 12345", confidence=0.75),
            ExtractedField(field_name="class", value="Class C", confidence=0.89),
            ExtractedField(field_name="issue_date", value="2019-08-15", confidence=0.78),
            ExtractedField(field_name="expiry_date", value="2027-08-15", confidence=0.80),
            ExtractedField(field_name="restrictions", value="NONE", confidence=0.85)
        ],
        total_fields_found=9
    )
    
    accuracy = AccuracyAssessment(
        overall_accuracy=0.86,
        field_accuracies={
            "name": 0.93, "surname": 0.96, "license_number": 0.91,
            "date_of_birth": 0.87, "address": 0.75, "class": 0.89,
            "issue_date": 0.78, "expiry_date": 0.80, "restrictions": 0.85
        },
        issues_found=[
            "Address field has lower confidence due to small font",
            "Issue and expiry dates may need manual verification"
        ],
        recommendations=[
            "Request higher resolution image for address clarity",
            "Implement date format validation",
            "Add manual review step for critical fields below 80% confidence"
        ]
    )
    
    decision = FinalDecision(
        accept=True,
        confidence=0.86,
        reasoning="Good classification confidence (88%) and acceptable extraction quality (86%). Most critical fields extracted successfully.",
        required_actions=["Manual verification recommended for address field"]
    )
    
    return ProcessingResult(
        classification=classification,
        extraction=extraction,
        accuracy_assessment=accuracy,
        final_decision=decision,
        processing_time=3.2
    )

def print_processing_results(result: ProcessingResult, processing_time: float):
    """Print formatted processing results."""
    
    print(f"Processing Time: {processing_time:.2f} seconds")
    print()
    
    # Classification Results
    print("📋 DOCUMENT CLASSIFICATION")
    print(f"Type: {result.classification.document_type.value.upper()}")
    print(f"Confidence: {result.classification.confidence:.2%}")
    print(f"Reasoning: {result.classification.reasoning}")
    print()
    
    # Extraction Results
    print("📝 FIELD EXTRACTION")
    print(f"Fields Found: {result.extraction.total_fields_found}")
    for field in result.extraction.extracted_fields:
        print(f"  • {field.field_name}: {field.value} (confidence: {field.confidence:.2%})")
    print()
    
    # Accuracy Assessment
    print("🎯 ACCURACY ASSESSMENT")
    print(f"Overall Accuracy: {result.accuracy_assessment.overall_accuracy:.2%}")
    print("Field Accuracies:")
    for field, accuracy in result.accuracy_assessment.field_accuracies.items():
        print(f"  • {field}: {accuracy:.2%}")
    
    if result.accuracy_assessment.issues_found:
        print("Issues Found:")
        for issue in result.accuracy_assessment.issues_found:
            print(f"  ⚠️  {issue}")
    
    if result.accuracy_assessment.recommendations:
        print("Recommendations:")
        for rec in result.accuracy_assessment.recommendations:
            print(f"  💡 {rec}")
    print()
    
    # Final Decision
    print("⚖️ FINAL DECISION")
    status = "✅ ACCEPTED" if result.final_decision.accept else "❌ REJECTED"
    print(f"Status: {status}")
    print(f"Confidence: {result.final_decision.confidence:.2%}")
    print(f"Reasoning: {result.final_decision.reasoning}")
    
    if result.final_decision.required_actions:
        print("Required Actions:")
        for action in result.final_decision.required_actions:
            print(f"  • {action}")
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(demo_langchain_processing())
