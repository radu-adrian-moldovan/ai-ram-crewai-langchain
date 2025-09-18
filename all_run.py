#!/usr/bin/env python3
"""
Comprehensive Demo for Progress Collector System
Processes all documents from all subdirectories in samples/input
Uses actual CrewAI and LangChain implementations.
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Union
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.main.core.progress_collector import (
    ProgressCollectorMCP, ImplementationType, EventType
)
from src.main.core.models import ProcessingResult
from src.main.core.manager_llm import get_llm_status
from src.main.core.manager_processor import DocumentProcessorBase, DocumentProcessorFactory
from src.main.core.manager_report import DocumentProcessingReportManager
import random

def display_processor_results(result: Dict[str, Any], processor_name: str):
    """Display JSON outputs for a processor result"""
    if result.get('status') == 'success':
        print(f"    📊 {processor_name} Results:")
        json_outputs = [
            ('🔍 Classification', 'classification_json'),
            ('📝 Extraction', 'extraction_json'),
            ('🎯 Accuracy', 'accuracy_json'),
            ('✅ Final Decision', 'final_decision_json')
        ]
        
        for label, key in json_outputs:
            if result.get(key):
                print(f"      {label}: {result[key]}")


def discover_all_documents(base_path: str) -> Dict[str, List[str]]:
    """Discover all documents organized by subdirectory"""
    base_dir = Path(base_path)
    documents_by_type = {}
    
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            doc_type = subdir.name
            documents = []
            for doc_file in subdir.iterdir():
                if doc_file.is_file() and doc_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.pdf']:
                    documents.append(str(doc_file))
            documents_by_type[doc_type] = sorted(documents)
    
    return documents_by_type


async def process_document_batch(documents: List[str], doc_type: str, 
                               crewai_processor: DocumentProcessorBase, 
                               langchain_processor: DocumentProcessorBase) -> Dict[str, List[Dict]]:
    """Process a batch of documents of the same type"""
    
    print(f"\n📁 Processing {doc_type.replace('_', ' ').title()} Documents ({len(documents)} files)")
    print("=" * 60)
    
    results = {'crewai': [], 'langchain': []}
    
    for i, doc_path in enumerate(documents, 1):
        doc_name = Path(doc_path).name
        print(f"  📄 {i}/{len(documents)}: {doc_name}")
        
        # Process with both processors in parallel
        print(f"\n\n\n    🚀 Processing with both CrewAI and LangChain in parallel...")
        print(f"    🚀 ------------------------------------------------------.")
        parallel_start = time.time()
        
        crewai_result, langchain_result = await asyncio.gather(
            crewai_processor.process(doc_path),
            langchain_processor.process(doc_path)
        )
        
        parallel_time = time.time() - parallel_start
        print(f"    ⚡ Parallel processing completed in {parallel_time:.2f}s")
        
        results['crewai'].append(crewai_result)
        results['langchain'].append(langchain_result)
        
        # Display results for both processors
        display_processor_results(crewai_result, "CrewAI")
        display_processor_results(langchain_result, "LangChain")
        
        # Brief pause between documents
        await asyncio.sleep(0.05)
    
    return results


async def main():

    print("🚀 CrewAI vs Langchain")
    print("🎯 Processing ALL documents from samples/input subdirectories")
    print("🤖 Using ACTUAL CrewAI and LangChain implementations")
    print("=" * 70)
    
    # Show LLM Manager Status
    print("\n🧠 LLM Manager Status:")
    print("-" * 30)
    llm_status = get_llm_status()
    print(f"  Current Provider: {llm_status['current_provider']}")
    print(f"  Ollama Available: {llm_status['ollama_available']}")
    print(f"  Ollama URL: {llm_status['ollama_url']}")
    print(f"  OpenAI Key Set: {llm_status['openai_key_set']}")
    print(f"  🌐 Offline Capable: {llm_status['offline_capable']}")
    
    if llm_status['offline_capable']:
        print("  ✅ OFFLINE MODE: Using local Ollama server")
    else:
        print("  ⚠️  ONLINE MODE: Will use internet-based APIs")
    
    # Initialize progress collector and report manager
    global_progress_collector = ProgressCollectorMCP()
    report_manager = DocumentProcessingReportManager(global_progress_collector)
    
    # Create processors using factory pattern
    print("📦 Initializing processors...")
    try:
        crewai_processor = DocumentProcessorFactory.build(ImplementationType.CREWAI, global_progress_collector)
        print("✅ CrewAI processor ready")
    except Exception as e:
        print(f"❌ CrewAI processor failed: {e}")
        return
    
    try:
        langchain_processor = DocumentProcessorFactory.build(ImplementationType.LANGCHAIN, global_progress_collector)
        print("✅ LangChain processor ready")
    except Exception as e:
        print(f"❌ LangChain processor failed: {e}")
        return
    
    # Discover all documents
    # input_base = "samples/input"
    # documents_by_type = discover_all_documents(input_base)
    
    documents_by_type = {
        # "passport": ["samples/input/passport/passport.png"],
        # "driverlicence": ["samples/input/driverlicence/driverlicence_back_ilie.png"],
        "idcard_classic": ["samples/input/idcard_classic/idcard_classic_JhonDoe.png"]
    }
    
    total_docs = sum(len(docs) for docs in documents_by_type.values())
    print(f"📊 Found {total_docs} documents across {len(documents_by_type)} document types:")
    for doc_type, docs in documents_by_type.items():
        print(f"  • {doc_type.replace('_', ' ').title()}: {len(docs)} files")
    
    print(f"\n⏱️  Estimated processing time: ~{total_docs * 2 * 1.2:.1f} seconds")
    print("🔄 Starting batch processing...")
    
    # Process each document type
    all_results = {}
    start_time = time.time()
    
    for doc_type, documents in documents_by_type.items():
        batch_results = await process_document_batch(
            documents, doc_type, crewai_processor, langchain_processor
        )
        all_results[doc_type] = batch_results
        
        # Show progress
        processed_docs = sum(len(results['crewai']) + len(results['langchain']) 
                           for results in all_results.values())
        total_operations = total_docs * 2
        progress_pct = (processed_docs / total_operations) * 100
        print(f"    ✅ {doc_type} completed | Overall progress: {progress_pct:.1f}%")
    
    processing_time = time.time() - start_time
    print(f"\n🎉 All processing completed in {processing_time:.2f} seconds!")
    
    # Generate complete analysis and reports using the report manager
    report_manager.generate_complete_analysis(
        all_results=all_results,
        total_docs=total_docs,
        processing_time=processing_time,
        documents_by_type=documents_by_type
    )


if __name__ == "__main__":
    asyncio.run(main())
