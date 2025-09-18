#!/usr/bin/env python3
"""
Comprehensive comparison demo between CrewAI and LangChain implementations
Run this from the project root directory: python src/full_demo.py
"""

import sys
import asyncio
import time
import os
import glob
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.main.crewai.document_processor import CrewAIDocumentProcessor
from src.main.langchain.document_processor import LangChainDocumentProcessor
from src.main.core.progress_collector import ProgressCollectorMCP, ImplementationType
from main.core.manager_processor import EnhancedCrewAIProcessor, EnhancedLangChainProcessor, BatchProcessor

def discover_sample_documents():
    """Discover all sample documents in the organized samples directory."""
    samples_dir = project_root / "samples" / "input"
    
    if not samples_dir.exists():
        print(f"❌ Samples directory not found: {samples_dir}")
        return {}
    
    documents = {
        "driver_license": [],
        "id_card": [],
        "passport": []
    }
    
    # Map directory names to document types
    dir_mapping = {
        "driverlicence": "driver_license",
        "idcard_classic": "id_card", 
        "passport": "passport"
    }
    
    for dir_name, doc_type in dir_mapping.items():
        doc_dir = samples_dir / dir_name
        if doc_dir.exists():
            # Find all image files
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                for file_path in doc_dir.glob(ext):
                    documents[doc_type].append(str(file_path))
    
    return documents

def print_sample_inventory():
    """Print an inventory of available sample documents."""
    print_banner("SAMPLE DOCUMENT INVENTORY")
    
    documents = discover_sample_documents()
    total_docs = sum(len(docs) for docs in documents.values())
    
    print(f"📊 Total Documents Available: {total_docs}")
    
    for doc_type, doc_list in documents.items():
        if doc_list:
            print(f"\n📄 {doc_type.replace('_', ' ').title()}: {len(doc_list)} files")
            for doc_path in doc_list[:3]:  # Show first 3 files
                filename = Path(doc_path).name
                print(f"   - {filename}")
            if len(doc_list) > 3:
                print(f"   ... and {len(doc_list) - 3} more")
    
    return documents

def print_banner(title):
    """Print a formatted banner."""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")

def print_results(title, result, processing_time):
    """Print processing results in a formatted way."""
    print(f"\n📋 {title}")
    print("-" * 40)
    print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
    
    if hasattr(result, 'classification'):
        print(f"📝 Document Type: {result.classification.document_type}")
        print(f"🎯 Classification Confidence: {result.classification.confidence:.2f}")
        print(f"💭 Reasoning: {result.classification.reasoning}")
        
        print(f"\n📊 Extraction Results:")
        print(f"   Fields Found: {result.extraction.total_fields_found}")
        for field in result.extraction.extracted_fields[:3]:  # Show first 3 fields
            print(f"   - {field.field_name}: {field.value} (confidence: {field.confidence:.2f})")
        
        print(f"\n🎯 Accuracy Assessment:")
        print(f"   Overall: {result.accuracy_assessment.overall_accuracy:.2f}")
        if result.accuracy_assessment.issues_found:
            print(f"   Issues: {', '.join(result.accuracy_assessment.issues_found[:2])}")
        
        print(f"\n✅ Final Decision: {'ACCEPT' if result.final_decision.accept else 'REJECT'}")
        print(f"   Confidence: {result.final_decision.confidence:.2f}")
        print(f"   Reasoning: {result.final_decision.reasoning}")
    else:
        print(f"Result: {result}")

async def compare_implementations():
    """Compare both implementations with sample document."""
    print_banner("DOCUMENT PROCESSING COMPARISON")
    
    # Sample document (we'll use a mock image for this demo)
    sample_image_path = "samples/id_card_sample.jpg"
    
    print(f"📄 Processing document: {sample_image_path}")
    print("ℹ️  Note: Using mock processing if actual dependencies are not available")
    
    # Initialize processors
    crewai_processor = CrewAIDocumentProcessor()
    langchain_processor = LangChainDocumentProcessor()
    
    # Test CrewAI implementation
    try:
        print_banner("CREWAI IMPLEMENTATION")
        start_time = time.time()
        crewai_result = await crewai_processor.process_document(sample_image_path)
        crewai_time = time.time() - start_time
        print_results("CrewAI Results", crewai_result, crewai_time)
    except Exception as e:
        print(f"❌ CrewAI Error: {e}")
        crewai_result = None
        crewai_time = 0
    
    # Test LangChain implementation
    try:
        print_banner("LANGCHAIN IMPLEMENTATION")
        start_time = time.time()
        langchain_result = await langchain_processor.process_document(sample_image_path)
        langchain_time = time.time() - start_time
        print_results("LangChain Results", langchain_result, langchain_time)
    except Exception as e:
        print(f"❌ LangChain Error: {e}")
        langchain_result = None
        langchain_time = 0
    
    # Performance comparison
    if crewai_result and langchain_result:
        print_banner("PERFORMANCE COMPARISON")
        print(f"⏱️  CrewAI Processing Time: {crewai_time:.2f}s")
        print(f"⏱️  LangChain Processing Time: {langchain_time:.2f}s")
        print(f"🏆 Faster Implementation: {'CrewAI' if crewai_time < langchain_time else 'LangChain'}")
        
        # Accuracy comparison
        crewai_accuracy = crewai_result.accuracy_assessment.overall_accuracy
        langchain_accuracy = langchain_result.accuracy_assessment.overall_accuracy
        print(f"🎯 CrewAI Accuracy: {crewai_accuracy:.2f}")
        print(f"🎯 LangChain Accuracy: {langchain_accuracy:.2f}")
        print(f"🏆 More Accurate: {'CrewAI' if crewai_accuracy > langchain_accuracy else 'LangChain'}")

async def batch_processing_demo():
    """Demonstrate batch processing capabilities with real document samples."""
    print_banner("BATCH PROCESSING DEMO")
    
    # Discover available documents
    documents = discover_sample_documents()
    total_docs = sum(len(docs) for docs in documents.values())
    
    if total_docs == 0:
        print("❌ No sample documents found in samples/input/")
        return
    
    print(f"📄 Processing {total_docs} documents across {len([k for k, v in documents.items() if v])} document types...")
    
    processors = {
        "CrewAI": CrewAIDocumentProcessor(),
        "LangChain": LangChainDocumentProcessor()
    }
    
    # Process each document type
    for doc_type, doc_list in documents.items():
        if not doc_list:
            continue
            
        print(f"\n🔄 Processing {doc_type.replace('_', ' ').title()} Documents ({len(doc_list)} files)")
        
        for proc_name, processor in processors.items():
            print(f"\n   🤖 {proc_name} Processing:")
            start_time = time.time()
            
            results = []
            successful = 0
            
            for i, doc_path in enumerate(doc_list, 1):
                try:
                    filename = Path(doc_path).name
                    print(f"      📄 {i}/{len(doc_list)}: {filename}")
                    result = await processor.process_document(doc_path)
                    results.append(result)
                    successful += 1
                    
                    # Show basic result info
                    if hasattr(result, 'classification'):
                        print(f"         ✅ Type: {result.classification.document_type} (confidence: {result.classification.confidence:.2f})")
                    else:
                        print(f"         ✅ Processed successfully")
                        
                except Exception as e:
                    print(f"         ❌ Error: {str(e)[:50]}...")
                    results.append(None)
            
            total_time = time.time() - start_time
            
            print(f"      📊 Results: {successful}/{len(doc_list)} successful")
            print(f"      ⏱️  Total Time: {total_time:.2f}s")
            print(f"      📈 Avg Time/Doc: {total_time/len(doc_list):.2f}s")

async def enhanced_comparison_demo():
    """Run enhanced comparison demo with progress collection and Excel reporting."""
    print_banner("ENHANCED DOCUMENT PROCESSING COMPARISON WITH PROGRESS TRACKING")
    
    # Initialize progress collector
    progress_collector = ProgressCollectorMCP()
    batch_processor = BatchProcessor(progress_collector)
    
    # Discover available documents
    documents = print_sample_inventory()
    
    if not any(documents.values()):
        print("❌ No sample documents found. Please add documents to samples/input/")
        return None
    
    # Collect all document paths for testing
    all_document_paths = []
    for doc_type, doc_list in documents.items():
        if doc_list:
            # Take first 3 documents from each type for demo
            all_document_paths.extend(doc_list[:3])
    
    if not all_document_paths:
        print("❌ No documents available for processing")
        return None
    
    print(f"\n🔄 Running enhanced comparison on {len(all_document_paths)} documents...")
    
    # Run comparison with progress tracking
    try:
        comparison_results = await batch_processor.compare_implementations(all_document_paths)
        
        # Print summary results
        print_banner("COMPARISON RESULTS SUMMARY")
        comp = comparison_results['comparison']
        
        print(f"📊 Documents Processed: {comp['document_count']}")
        print(f"\n🤖 CrewAI Results:")
        print(f"   ✅ Successful: {comp['crewai']['successful']}/{comp['document_count']}")
        print(f"   ❌ Failed: {comp['crewai']['failed']}")
        print(f"   📈 Success Rate: {comp['crewai']['success_rate']:.1%}")
        print(f"   ⏱️  Total Time: {comp['crewai']['total_time']:.2f}s")
        print(f"   📊 Avg Time/Doc: {comp['crewai']['avg_time_per_doc']:.2f}s")
        
        print(f"\n🦜 LangChain Results:")
        print(f"   ✅ Successful: {comp['langchain']['successful']}/{comp['document_count']}")
        print(f"   ❌ Failed: {comp['langchain']['failed']}")
        print(f"   📈 Success Rate: {comp['langchain']['success_rate']:.1%}")
        print(f"   ⏱️  Total Time: {comp['langchain']['total_time']:.2f}s")
        print(f"   📊 Avg Time/Doc: {comp['langchain']['avg_time_per_doc']:.2f}s")
        
        print(f"\n🏆 Performance Winners:")
        print(f"   ⚡ Faster: {comp['faster_implementation']} (by {comp['time_difference']:.2f}s)")
        print(f"   🛡️  More Reliable: {comp['more_reliable']}")
        
        # Generate Excel report
        print_banner("GENERATING EXCEL REPORT")
        print("📊 Creating comprehensive Excel report with detailed metrics...")
        
        # Save report to samples/output directory
        output_dir = project_root / "samples" / "output"
        output_dir.mkdir(exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"document_processing_comparison_report_{timestamp}.xlsx"
        
        generated_path = progress_collector.generate_excel_report(str(report_path))
        
        print(f"✅ Excel report generated: {generated_path}")
        print(f"📋 Report includes:")
        print(f"   - Executive Summary with key metrics")
        print(f"   - Detailed event log for both implementations")
        print(f"   - Performance metrics comparison")
        print(f"   - Visual charts and analysis")
        
        # Print final statistics
        stats = progress_collector.get_summary_statistics()
        print_banner("SESSION STATISTICS")
        print(f"📊 Session ID: {stats['session_info']['session_id']}")
        print(f"⏱️  Session Duration: {(datetime.now() - stats['session_info']['start_time']).total_seconds():.2f}s")
        print(f"📈 Total Events Recorded: {stats['session_info']['total_events']}")
        print(f"🤖 CrewAI Events: {len(progress_collector.collector.events[ImplementationType.CREWAI])}")
        print(f"🦜 LangChain Events: {len(progress_collector.collector.events[ImplementationType.LANGCHAIN])}")
        
        return {
            'progress_collector': progress_collector,
            'comparison_results': comparison_results,
            'report_path': generated_path
        }
        
    except Exception as e:
        print(f"❌ Enhanced comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def quick_single_document_demo():
    """Quick demo with a single document from each type."""
    print_banner("QUICK SINGLE DOCUMENT DEMO")
    
    # Initialize progress collector
    progress_collector = ProgressCollectorMCP()
    
    # Create enhanced processors
    crewai_processor = EnhancedCrewAIProcessor(progress_collector)
    langchain_processor = EnhancedLangChainProcessor(progress_collector)
    
    # Discover documents
    documents = discover_sample_documents()
    
    # Pick one document from each type
    test_documents = []
    for doc_type, doc_list in documents.items():
        if doc_list:
            test_documents.append((doc_type, doc_list[0]))
    
    if not test_documents:
        print("❌ No test documents available")
        return None
    
    print(f"🔍 Testing {len(test_documents)} representative documents...")
    
    for doc_type, doc_path in test_documents:
        filename = Path(doc_path).name
        expected_type = doc_type.replace('_', ' ').title()
        
        print(f"\n📄 Processing: {filename} (Expected: {expected_type})")
        
        # Test both implementations
        try:
            print("   🤖 CrewAI processing...")
            start_time = time.time()
            crewai_result = await crewai_processor.process_document(doc_path)
            crewai_time = time.time() - start_time
            
            print("   🦜 LangChain processing...")
            start_time = time.time()
            langchain_result = await langchain_processor.process_document(doc_path)
            langchain_time = time.time() - start_time
            
            # Quick comparison
            print(f"   ⚡ Speed: CrewAI {crewai_time:.2f}s vs LangChain {langchain_time:.2f}s")
            
            if hasattr(crewai_result, 'classification') and hasattr(langchain_result, 'classification'):
                crewai_type = crewai_result.classification.document_type
                langchain_type = langchain_result.classification.document_type
                crewai_conf = crewai_result.classification.confidence
                langchain_conf = langchain_result.classification.confidence
                
                print(f"   📋 Classification: CrewAI={crewai_type}({crewai_conf:.2f}) vs LangChain={langchain_type}({langchain_conf:.2f})")
                
                # Check accuracy
                expected_type_normalized = doc_type.lower()
                crewai_correct = crewai_type.lower() == expected_type_normalized
                langchain_correct = langchain_type.lower() == expected_type_normalized
                
                print(f"   🎯 Accuracy: CrewAI={'✅' if crewai_correct else '❌'} LangChain={'✅' if langchain_correct else '❌'}")
        
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
    
    # Generate quick report
    output_dir = project_root / "samples" / "output"
    output_dir.mkdir(exist_ok=True)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"quick_demo_report_{timestamp}.xlsx"
    
    generated_path = progress_collector.generate_excel_report(str(report_path))
    print(f"\n📊 Quick demo report generated: {generated_path}")
    
    return {
        'progress_collector': progress_collector,
        'report_path': generated_path
    }

async def document_type_accuracy_test():
    """Test classification accuracy across different document types."""
    print_banner("DOCUMENT TYPE ACCURACY TEST")
    
    documents = discover_sample_documents()
    processors = {
        "CrewAI": CrewAIDocumentProcessor(),
        "LangChain": LangChainDocumentProcessor()
    }
    
    # Expected document types based on folder structure
    expected_types = {
        "driver_license": "driver_license",
        "id_card": "id_card", 
        "passport": "passport"
    }
    
    accuracy_results = {}
    
    for proc_name, processor in processors.items():
        print(f"\n🎯 Testing {proc_name} Classification Accuracy:")
        
        correct_classifications = 0
        total_documents = 0
        type_accuracy = {}
        
        for doc_type, doc_list in documents.items():
            if not doc_list:
                continue
                
            expected_type = expected_types.get(doc_type, doc_type)
            type_correct = 0
            
            print(f"\n   📂 {doc_type.replace('_', ' ').title()}: {len(doc_list)} documents")
            
            for doc_path in doc_list[:5]:  # Test first 5 of each type
                try:
                    filename = Path(doc_path).name
                    result = await processor.process_document(doc_path)
                    total_documents += 1
                    
                    if hasattr(result, 'classification'):
                        classified_type = result.classification.document_type
                        confidence = result.classification.confidence
                        
                        if classified_type.lower() == expected_type.lower():
                            correct_classifications += 1
                            type_correct += 1
                            status = "✅"
                        else:
                            status = "❌"
                        
                        print(f"      {status} {filename}: {classified_type} (conf: {confidence:.2f})")
                    else:
                        print(f"      ⚠️  {filename}: No classification result")
                        
                except Exception as e:
                    print(f"      ❌ {filename}: Error - {str(e)[:30]}...")
                    total_documents += 1
            
            if len(doc_list) > 0:
                type_accuracy[doc_type] = type_correct / min(len(doc_list), 5)
        
        overall_accuracy = correct_classifications / total_documents if total_documents > 0 else 0
        accuracy_results[proc_name] = {
            'overall': overall_accuracy,
            'by_type': type_accuracy,
            'total_docs': total_documents
        }
        
        print(f"\n   📊 {proc_name} Results:")
        print(f"      Overall Accuracy: {overall_accuracy:.2%}")
        for doc_type, acc in type_accuracy.items():
            print(f"      {doc_type.replace('_', ' ').title()}: {acc:.2%}")
    
    # Compare results
    if len(accuracy_results) > 1:
        print_banner("ACCURACY COMPARISON")
        for proc_name, results in accuracy_results.items():
            print(f"🎯 {proc_name}: {results['overall']:.2%} overall accuracy ({results['total_docs']} docs tested)")

async def compare_implementations():
    """Compare both implementations with sample documents."""
    print_banner("DOCUMENT PROCESSING COMPARISON")
    
    # First show what's available
    documents = print_sample_inventory()
    
    if not any(documents.values()):
        print("❌ No sample documents found. Please add documents to samples/input/")
        return
    
    # Pick one sample from each type for detailed comparison
    sample_docs = []
    for doc_type, doc_list in documents.items():
        if doc_list:
            sample_docs.append((doc_type, doc_list[0]))
    
    if not sample_docs:
        print("❌ No sample documents available for comparison")
        return
    
    print(f"\n📄 Running detailed comparison on {len(sample_docs)} sample documents...")
    
    # Initialize processors
    crewai_processor = CrewAIDocumentProcessor()
    langchain_processor = LangChainDocumentProcessor()
    
    for doc_type, doc_path in sample_docs:
        filename = Path(doc_path).name
        print(f"\n📋 Processing: {filename} (Expected: {doc_type.replace('_', ' ').title()})")
        
        # Test CrewAI implementation
        try:
            print(f"\n🤖 CrewAI Processing:")
            start_time = time.time()
            crewai_result = await crewai_processor.process_document(doc_path)
            crewai_time = time.time() - start_time
            print_results("CrewAI Results", crewai_result, crewai_time)
        except Exception as e:
            print(f"❌ CrewAI Error: {e}")
            crewai_result = None
            crewai_time = 0
        
        # Test LangChain implementation
        try:
            print(f"\n🦜 LangChain Processing:")
            start_time = time.time()
            langchain_result = await langchain_processor.process_document(doc_path)
            langchain_time = time.time() - start_time
            print_results("LangChain Results", langchain_result, langchain_time)
        except Exception as e:
            print(f"❌ LangChain Error: {e}")
            langchain_result = None
            langchain_time = 0
        
        # Quick comparison for this document
        if crewai_result and langchain_result:
            print(f"\n⚡ Quick Comparison for {filename}:")
            print(f"   ⏱️  Speed: CrewAI {crewai_time:.2f}s vs LangChain {langchain_time:.2f}s")
            
            if hasattr(crewai_result, 'classification') and hasattr(langchain_result, 'classification'):
                print(f"   🎯 Classification: CrewAI={crewai_result.classification.document_type} vs LangChain={langchain_result.classification.document_type}")
                print(f"   📊 Confidence: CrewAI={crewai_result.classification.confidence:.2f} vs LangChain={langchain_result.classification.confidence:.2f}")

async def batch_processing_demo_old():
    """Demonstrate batch processing capabilities."""
    print_banner("BATCH PROCESSING DEMO")
    
    # Sample documents for batch processing
    sample_docs = [
        "samples/id_card_sample.jpg",
        "samples/driver_license_sample.jpg",
        "samples/passport_sample.jpg"
    ]
    
    print(f"📄 Processing {len(sample_docs)} documents...")
    
    processors = {
        "CrewAI": CrewAIDocumentProcessor(),
        "LangChain": LangChainDocumentProcessor()
    }
    
    for proc_name, processor in processors.items():
        print(f"\n🔄 {proc_name} Batch Processing:")
        start_time = time.time()
        
        results = []
        for i, doc_path in enumerate(sample_docs, 1):
            try:
                print(f"   Processing document {i}/{len(sample_docs)}...")
                result = await processor.process_document(doc_path)
                results.append(result)
            except Exception as e:
                print(f"   ❌ Error processing document {i}: {e}")
                results.append(None)
        
        total_time = time.time() - start_time
        successful = len([r for r in results if r is not None])
        
        print(f"   ✅ Completed: {successful}/{len(sample_docs)} documents")
        print(f"   ⏱️  Total Time: {total_time:.2f}s")
        print(f"   📊 Avg Time/Doc: {total_time/len(sample_docs):.2f}s")

def main():
    """Main function to run the comprehensive document processing demo with progress tracking."""
    try:
        print_banner("COMPREHENSIVE DOCUMENT PROCESSING DEMO WITH PROGRESS TRACKING")
        print("🚀 Testing both CrewAI and LangChain implementations with real document samples")
        print("📊 Full progress tracking and Excel report generation enabled")
        
        # Check if pandas and openpyxl are available
        try:
            import pandas as pd
            import openpyxl
            print("✅ Excel reporting dependencies available")
        except ImportError as e:
            print(f"⚠️  Excel reporting dependencies missing: {e}")
            print("   Installing required packages...")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "pandas", "openpyxl"], check=True)
            print("✅ Dependencies installed successfully")
        
        print("\n" + "="*60)
        print("🎯 DEMO OPTIONS:")
        print("1. Enhanced Full Comparison Demo (recommended)")
        print("2. Quick Single Document Demo")
        print("3. Legacy Comparison Demo")
        print("="*60)
        
        # For automation, we'll run the enhanced demo
        # In interactive mode, you could add input() here
        choice = "1"  # Default to enhanced demo
        
        if choice == "1":
            print("\n🚀 Running Enhanced Full Comparison Demo...")
            result = asyncio.run(enhanced_comparison_demo())
            
            if result:
                print("\n✅ Enhanced demo completed successfully!")
                print(f"📊 Excel report available at: {result['report_path']}")
                print("\n💡 Next steps:")
                print("   - Open the Excel report to view detailed metrics")
                print("   - Review the 'Executive Summary' sheet for key insights")
                print("   - Check 'Detailed Events' for complete processing logs")
                
        elif choice == "2":
            print("\n🚀 Running Quick Single Document Demo...")
            result = asyncio.run(quick_single_document_demo())
            
            if result:
                print("\n✅ Quick demo completed successfully!")
                print(f"📊 Report available at: {result['report_path']}")
        
        else:
            print("\n🚀 Running Legacy Comparison Demo...")
            # Run original demos
            print_sample_inventory()
            asyncio.run(compare_implementations())
            asyncio.run(document_type_accuracy_test())
            asyncio.run(batch_processing_demo())
        
        print_banner("DEMO COMPLETED SUCCESSFULLY")
        print("✅ All tests completed!")
        print("\n📊 Summary:")
        print("   - Progress tracking system implemented")
        print("   - Event-driven architecture deployed")
        print("   - Comprehensive Excel reports generated")
        print("   - Performance metrics collected and analyzed")
        print("\n💡 Tips:")
        print("   - Excel reports contain detailed comparison metrics")
        print("   - Check samples/output/ for generated reports")
        print("   - Use progress collector for continuous monitoring")
        print("   - Customize metrics and thresholds as needed")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
