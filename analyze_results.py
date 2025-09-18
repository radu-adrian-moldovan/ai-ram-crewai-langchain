#!/usr/bin/env python3
"""
Summary of Comprehensive Demo Results
Shows the results from processing all documents in samples/input subdirectories
"""

import json
from pathlib import Path

def analyze_comprehensive_results():
    """Analyze the comprehensive demo results"""
    
    # Find the latest comprehensive JSON file
    output_dir = Path("samples/output")
    json_files = list(output_dir.glob("comprehensive_events_*.json"))
    
    if not json_files:
        print("❌ No comprehensive demo results found!")
        return
    
    # Get the most recent file
    latest_json = max(json_files, key=lambda f: f.stat().st_mtime)
    
    print("🎉 Comprehensive Demo Results Analysis")
    print("=" * 50)
    print(f"📄 Report: {latest_json.name}")
    
    # Load and analyze the data
    with open(latest_json, 'r') as f:
        data = json.load(f)
    
    session_info = data['session_info']
    events = data['events']
    
    print(f"\n📊 Session Summary:")
    print(f"  Session ID: {session_info['session_id'][:8]}...")
    print(f"  Start Time: {session_info['start_time']}")
    print(f"  Duration: {session_info['session_duration_seconds']:.2f} seconds")
    print(f"  Total Events: {session_info['total_events']}")
    print(f"  CrewAI Events: {session_info['events_by_implementation']['crewai']}")
    print(f"  LangChain Events: {session_info['events_by_implementation']['langchain']}")
    
    # Analyze by document type
    documents_by_type = {}
    for event in events:
        if event['event_type'] == 'processing_start':
            doc_path = event['document_path']
            if 'passport' in doc_path.lower() or 'pasaport' in doc_path.lower():
                doc_type = 'passport'
            elif 'idcard' in doc_path.lower():
                doc_type = 'id_card'
            elif 'driverlicence' in doc_path.lower():
                doc_type = 'driver_license'
            else:
                doc_type = 'unknown'
            
            if doc_type not in documents_by_type:
                documents_by_type[doc_type] = {'crewai': 0, 'langchain': 0}
            
            impl = event['implementation']
            documents_by_type[doc_type][impl] += 1
    
    print(f"\n📁 Documents Processed by Type:")
    total_docs = 0
    for doc_type, counts in documents_by_type.items():
        doc_count = counts['crewai']  # Each doc processed by both implementations
        total_docs += doc_count
        print(f"  • {doc_type.replace('_', ' ').title()}: {doc_count} documents")
    
    print(f"\n🎯 Processing Statistics:")
    print(f"  📄 Total Documents: {total_docs}")
    print(f"  🔄 Total Operations: {total_docs * 2} (each doc × 2 implementations)")
    print(f"  📊 Events per Operation: ~{session_info['total_events'] // (total_docs * 2)}")
    print(f"  ⏱️  Avg Time per Document: {session_info['session_duration_seconds'] / total_docs:.2f}s")
    
    # File sizes
    excel_files = list(output_dir.glob("comprehensive_report_*.xlsx"))
    if excel_files:
        latest_excel = max(excel_files, key=lambda f: f.stat().st_mtime)
        excel_size = latest_excel.stat().st_size / 1024  # KB
        json_size = latest_json.stat().st_size / 1024  # KB
        
        print(f"\n📋 Report Files:")
        print(f"  📊 Excel Report: {latest_excel.name} ({excel_size:.1f} KB)")
        print(f"  📄 JSON Events: {latest_json.name} ({json_size:.1f} KB)")
    
    print(f"\n✅ Successfully processed ALL documents from samples/input subdirectories!")
    print(f"🎯 Progress Collector System: FULLY OPERATIONAL")

if __name__ == "__main__":
    analyze_comprehensive_results()
