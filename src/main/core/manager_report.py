#!/usr/bin/env python3
"""
Report Manager for Document Processing Results
Handles comprehensive session summaries, performance analysis, and report generation
"""

import time
from pathlib import Path
from typing import Dict, List, Any
from src.main.core.progress_collector import ProgressCollectorMCP


class DocumentProcessingReportManager:
    """Manager for generating comprehensive reports and analysis"""
    
    def __init__(self, progress_collector: ProgressCollectorMCP):
        self.progress_collector = progress_collector
    
    def display_session_summary(self):
        """Display comprehensive session summary"""
        print("\n📈 Comprehensive Session Summary:")
        print("=" * 50)
        summary = self.progress_collector.collector.get_session_summary()
        print(f"  Session ID: {summary['session_id'][:8]}...")
        print(f"  Total Events: {summary['total_events']}")
        print(f"  CrewAI Events: {summary['events_by_implementation'].get('crewai', 0)}")
        print(f"  LangChain Events: {summary['events_by_implementation'].get('langchain', 0)}")
        print(f"  Session Duration: {summary['session_duration_seconds']:.2f}s")
        return summary
    
    def analyze_performance_by_document_type(self, all_results: Dict[str, Any]):
        """Analyze and display performance results by document type"""
        print("\n📊 Performance Analysis by Document Type:")
        print("=" * 50)
        
        for doc_type, results in all_results.items():
            print(f"\n📁 {doc_type.replace('_', ' ').title()}:")
            
            # Filter successful results only
            successful_crewai = [r for r in results['crewai'] if r.get('status') == 'success']
            successful_langchain = [r for r in results['langchain'] if r.get('status') == 'success']
            
            if successful_crewai and successful_langchain:
                crewai_times = [r.get('total_processing_time', 0) for r in successful_crewai]
                langchain_times = [r.get('total_processing_time', 0) for r in successful_langchain]
                
                crewai_accuracies = [r.get('overall_accuracy', 0) for r in successful_crewai]
                langchain_accuracies = [r.get('overall_accuracy', 0) for r in successful_langchain]
                
                crewai_avg_time = sum(crewai_times) / len(crewai_times)
                langchain_avg_time = sum(langchain_times) / len(langchain_times)
                
                crewai_avg_accuracy = sum(crewai_accuracies) / len(crewai_accuracies)
                langchain_avg_accuracy = sum(langchain_accuracies) / len(langchain_accuracies)
                
                print(f"  ⏱️  Avg Processing Time:")
                print(f"     CrewAI: {crewai_avg_time:.3f}s | LangChain: {langchain_avg_time:.3f}s")
                speed_winner = "CrewAI" if crewai_avg_time < langchain_avg_time else "LangChain"
                print(f"     🏆 Faster: {speed_winner}")
                
                print(f"  🎯 Avg Accuracy:")
                print(f"     CrewAI: {crewai_avg_accuracy:.3f} | LangChain: {langchain_avg_accuracy:.3f}")
                accuracy_winner = "CrewAI" if crewai_avg_accuracy > langchain_avg_accuracy else "LangChain"
                print(f"     🏆 More Accurate: {accuracy_winner}")
            else:
                print(f"  ⚠️  Insufficient successful results for comparison")
                if not successful_crewai:
                    print(f"     CrewAI: {len(results['crewai'])} failed")
                if not successful_langchain:
                    print(f"     LangChain: {len(results['langchain'])} failed")
    
    def generate_comprehensive_reports(self, output_dir: str = "samples/output") -> Dict[str, str]:
        """Generate comprehensive reports (Excel and JSON)"""
        print("\n📋 Generating Comprehensive Reports...")
        print("=" * 40)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate timestamped filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Generate Excel report
        excel_file = self.progress_collector.generate_excel_report(
            str(output_path / f"comprehensive_report_{timestamp}.xlsx")
        )
        print(f"  ✅ Excel Report: {Path(excel_file).name}")
        
        # Generate JSON report
        json_file = self.progress_collector.export_events_json(
            str(output_path / f"comprehensive_events_{timestamp}.json")
        )
        print(f"  ✅ JSON Events: {Path(json_file).name}")
        
        return {
            'excel_file': excel_file,
            'json_file': json_file,
            'output_dir': str(output_path)
        }
    
    def display_final_summary(self, total_docs: int, processing_time: float, 
                            documents_by_type: Dict[str, List[str]], summary: Dict[str, Any]):
        """Display final summary statistics"""
        print(f"\n📈 Final Summary:")
        print(f"  📄 Documents Processed: {total_docs}")
        print(f"  🔄 Total Operations: {total_docs * 2}")
        print(f"  📊 Events Logged: {summary['total_events']}")
        print(f"  ⏱️  Total Time: {processing_time:.2f}s")
        print(f"  📁 Document Types: {len(documents_by_type)}")
    
    def analyze_overall_winners(self, all_results: Dict[str, Any]):
        """Analyze and display overall winner comparison"""
        # Overall winner analysis
        all_crewai_times = []
        all_langchain_times = []
        all_crewai_accuracies = []
        all_langchain_accuracies = []
        
        for results in all_results.values():
            # Filter out error results
            successful_crewai = [r for r in results['crewai'] if r.get('status') == 'success']
            successful_langchain = [r for r in results['langchain'] if r.get('status') == 'success']
            
            all_crewai_times.extend([r.get('total_processing_time', 0) for r in successful_crewai])
            all_langchain_times.extend([r.get('total_processing_time', 0) for r in successful_langchain])
            all_crewai_accuracies.extend([r.get('overall_accuracy', 0) for r in successful_crewai])
            all_langchain_accuracies.extend([r.get('overall_accuracy', 0) for r in successful_langchain])
        
        if all_crewai_times and all_langchain_times:
            overall_crewai_time = sum(all_crewai_times) / len(all_crewai_times)
            overall_langchain_time = sum(all_langchain_times) / len(all_langchain_times)
            overall_crewai_accuracy = sum(all_crewai_accuracies) / len(all_crewai_accuracies)
            overall_langchain_accuracy = sum(all_langchain_accuracies) / len(all_langchain_accuracies)
            
            print(f"\n🏆 Overall Winner Analysis:")
            print(f"  ⚡ Speed: {'CrewAI' if overall_crewai_time < overall_langchain_time else 'LangChain'}")
            print(f"  🎯 Accuracy: {'CrewAI' if overall_crewai_accuracy > overall_langchain_accuracy else 'LangChain'}")
        else:
            print(f"\n⚠️  Insufficient successful results for winner analysis")
    
    def generate_complete_analysis(self, all_results: Dict[str, Any], total_docs: int, 
                                 processing_time: float, documents_by_type: Dict[str, List[str]], 
                                 output_dir: str = "samples/output") -> Dict[str, Any]:
        """Generate complete analysis and reports in one call"""
        
        # Display session summary
        summary = self.display_session_summary()
        
        # Analyze performance by document type
        self.analyze_performance_by_document_type(all_results)
        
        # Generate reports
        report_files = self.generate_comprehensive_reports(output_dir)
        
        # Display final summary
        self.display_final_summary(total_docs, processing_time, documents_by_type, summary)
        
        # Analyze overall winners
        self.analyze_overall_winners(all_results)
        
        # Final message
        print(f"\n📄 Reports saved to: {report_files['output_dir']}")
        print("🎉 Comprehensive demo completed successfully!")
        
        return {
            'summary': summary,
            'report_files': report_files,
            'processing_stats': {
                'total_docs': total_docs,
                'processing_time': processing_time,
                'document_types': len(documents_by_type)
            }
        }
