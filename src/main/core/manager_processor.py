#!/usr/bin/env python3
"""
Enhanced Document Processors with Progress Tracking Integration
Wraps the original processors to add progress collection and event tracking
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.main.crewai.document_processor import CrewAIDocumentProcessor
from src.main.langchain.document_processor import LangChainDocumentProcessor
from src.main.core.progress_collector import ProgressCollectorMCP, ImplementationType


class DocumentProcessorBase:
    """Base class for all document processors with common functionality"""
    
    def __init__(self, processor: Any, implementation_type: ImplementationType):
        self.processor = processor
        self.implementation_type = implementation_type
        self.name = implementation_type.value.title()
        self.progress_collector: Optional[ProgressCollectorMCP] = None
    
    def _save_trace_file(self, trace_data: Dict[str, Any], document_path: str):
        """Save individual trace file for this processing session"""
        try:
            # Create output directory
            output_dir = Path("samples/output/traces")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            doc_name = Path(document_path).stem
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            trace_filename = f"{self.implementation_type.value}_{doc_name}_{timestamp}.json"
            trace_path = output_dir / trace_filename
            
            # Enhanced trace data with complete route information
            enhanced_trace = {
                "processing_session": {
                    "session_id": f"session_{timestamp}_{self.implementation_type.value}",
                    "framework": self.implementation_type.value,
                    "document_path": document_path,
                    "start_time": trace_data.get('start_time', time.strftime("%Y-%m-%dT%H:%M:%SZ")),
                    "end_time": trace_data.get('end_time', time.strftime("%Y-%m-%dT%H:%M:%SZ")),
                    "total_processing_time_ms": trace_data.get('total_processing_time_ms', 0)
                },
                "processing_pipeline": trace_data.get('processing_pipeline', []),
                "step_details": trace_data.get('step_details', {}),
                "final_results": {
                    "classification": trace_data.get('classification_json'),
                    "extraction": trace_data.get('extraction_json'),
                    "accuracy": trace_data.get('accuracy_json'),
                    "final_decision": trace_data.get('final_decision_json')
                },
                "performance_metrics": {
                    "total_time": trace_data.get('total_processing_time', 0.0),
                    "framework_time": trace_data.get('framework_processing_time', 0.0),
                    "step_timings": trace_data.get('timing_details', {}),
                    "success_rate": trace_data.get('success_rate', 1.0)
                },
                "quality_assessment": {
                    "overall_accuracy": trace_data.get('overall_accuracy', 0.0),
                    "confidence": trace_data.get('confidence', 0.0),
                    "document_type": trace_data.get('document_type', 'unknown'),
                    "processing_status": trace_data.get('status', 'unknown')
                },
                "export_metadata": {
                    "export_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "trace_version": "2.1.0",
                    "processor_version": "enhanced"
                }
            }
            
            # Save trace data
            with open(trace_path, 'w', encoding='utf-8') as f:
                json.dump(enhanced_trace, f, indent=2, ensure_ascii=False)
            
            print(f"    💾 Enhanced trace saved: {trace_filename}")
            return str(trace_path)
        except Exception as e:
            print(f"    ⚠️  Failed to save trace: {e}")
            return None
    
    async def process(self, document_path: str, progress_collector: Optional[ProgressCollectorMCP] = None):
        """Process document using actual implementation and log to progress collector"""
        if progress_collector is None:
            # For enhanced processors that have their own progress collector
            if hasattr(self, 'progress_collector') and getattr(self, 'progress_collector') is not None:
                progress_collector = getattr(self, 'progress_collector')
            else:
                raise ValueError("Progress collector must be provided or set as instance attribute")
        
        # At this point progress_collector is guaranteed to be not None
        assert progress_collector is not None
        
        # Start processing
        start_time = time.time()
        processing_id = progress_collector.start_processing(
            self.implementation_type, 
            document_path,
            metadata={'document_path': document_path}
        )
        
        try:
            # Call the actual processor
            result = await self.processor.process_document(document_path)
            
            # Capture detailed step information
            step_details = {}
            timing_details = {}
            
            # Extract classification details
            classification_json = None
            if hasattr(result, 'classification') and result.classification:
                classification_json = {
                    "classification": {
                        "label": result.classification.document_type.value,
                        "confidence": result.classification.confidence
                    },
                    "metadata": {
                        "image_path": document_path,
                        "processing_time": getattr(result.classification, 'processing_time', 0.0)
                    }
                }
                
                progress_collector.record_classification(
                    self.implementation_type,
                    document_path,
                    result.classification.document_type.value,
                    result.classification.confidence,
                    f"Classified as {result.classification.document_type.value}"
                )
                
                step_details['classification'] = classification_json
                timing_details['classification_time'] = getattr(result.classification, 'processing_time', 0.0)
            
            # Extract extraction details
            extraction_json = None
            if hasattr(result, 'extraction') and result.extraction:
                extraction_fields = []
                extraction_data = {}
                
                if result.extraction.extracted_fields:
                    for field in result.extraction.extracted_fields:
                        extraction_fields.append({
                            "name": field.field_name,
                            "value": field.value
                        })
                        extraction_data[field.field_name] = field.value
                
                extraction_json = {
                    "fields": extraction_fields,
                    "metadata": {
                        "processing_time": getattr(result.extraction, 'processing_time', 0.0)
                    }
                }
                
                progress_collector.record_extraction(
                    self.implementation_type,
                    document_path,
                    extraction_data
                )
                
                step_details['extraction'] = extraction_json
                timing_details['extraction_time'] = getattr(result.extraction, 'processing_time', 0.0)
            
            # Extract accuracy details
            accuracy_json = None
            if hasattr(result, 'accuracy_assessment') and result.accuracy_assessment:
                accuracy_json = {
                    "classification": result.accuracy_assessment.overall_accuracy,
                    "extraction": result.accuracy_assessment.field_accuracies or {},
                    "overall": result.accuracy_assessment.overall_accuracy,
                    "metadata": {
                        "processing_time": getattr(result.accuracy_assessment, 'processing_time', 0.0)
                    }
                }
                
                accuracy_data = {
                    'overall_accuracy': result.accuracy_assessment.overall_accuracy,
                    'field_accuracies': result.accuracy_assessment.field_accuracies or {}
                }
                progress_collector.record_accuracy_assessment(
                    self.implementation_type,
                    document_path,
                    accuracy_data
                )
                
                step_details['accuracy'] = accuracy_json
                timing_details['accuracy_time'] = getattr(result.accuracy_assessment, 'processing_time', 0.0)
            
            # Calculate total processing time
            total_time = time.time() - start_time
            
            # Build processing pipeline for trace route
            processing_pipeline = []
            current_time = start_time
            
            if classification_json:
                step_duration = timing_details.get('classification_time', 0.0) * 1000  # Convert to ms
                processing_pipeline.append({
                    "step": "classification",
                    "step_number": 1,
                    "agent": f"{self.implementation_type.value}_classification_agent",
                    "start_time": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(current_time)),
                    "end_time": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(current_time + timing_details.get('classification_time', 0.0))),
                    "duration_ms": step_duration,
                    "status": "completed",
                    "result_summary": f"Classified as {classification_json.get('classification', {}).get('label', 'unknown')} with confidence {classification_json.get('classification', {}).get('confidence', 0.0)}"
                })
                current_time += timing_details.get('classification_time', 0.0)
            
            if extraction_json:
                step_duration = timing_details.get('extraction_time', 0.0) * 1000  # Convert to ms
                fields_count = len(extraction_json.get('fields', []))
                processing_pipeline.append({
                    "step": "extraction",
                    "step_number": 2,
                    "agent": f"{self.implementation_type.value}_extraction_agent",
                    "start_time": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(current_time)),
                    "end_time": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(current_time + timing_details.get('extraction_time', 0.0))),
                    "duration_ms": step_duration,
                    "status": "completed",
                    "result_summary": f"Extracted {fields_count} fields"
                })
                current_time += timing_details.get('extraction_time', 0.0)
            
            if accuracy_json:
                step_duration = timing_details.get('accuracy_time', 0.0) * 1000  # Convert to ms
                overall_acc = accuracy_json.get('overall', 0.0)
                processing_pipeline.append({
                    "step": "accuracy_assessment",
                    "step_number": 3,
                    "agent": f"{self.implementation_type.value}_accuracy_agent",
                    "start_time": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(current_time)),
                    "end_time": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(current_time + timing_details.get('accuracy_time', 0.0))),
                    "duration_ms": step_duration,
                    "status": "completed",
                    "result_summary": f"Overall accuracy {overall_acc:.2f}"
                })
                current_time += timing_details.get('accuracy_time', 0.0)
            
            # Add final decision step
            final_decision_time = total_time - sum(timing_details.values())
            final_step_duration = max(final_decision_time * 1000, 100)  # Minimum 100ms
            processing_pipeline.append({
                "step": "final_decision",
                "step_number": 4,
                "agent": f"{self.implementation_type.value}_final_decision_agent",
                "start_time": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(current_time)),
                "end_time": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(start_time + total_time)),
                "duration_ms": final_step_duration,
                "status": "completed",
                "result_summary": f"Processing completed successfully"
            })
            
            # Create final decision JSON
            final_decision_json = {
                "document_path": document_path,
                "implementation": self.implementation_type.value,
                "classification": classification_json.get("classification") if classification_json else None,
                "extraction": extraction_json.get("fields") if extraction_json else None,
                "accuracy": accuracy_json if accuracy_json else None,
                "timing": {
                    "total_processing_time": total_time,
                    "individual_steps": timing_details,
                    "framework_processing_time": result.processing_time if hasattr(result, 'processing_time') else 0.0
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processing_pipeline": processing_pipeline,
                "complete_trace_route": {
                    "processing_pipeline": processing_pipeline,
                    "total_processing_time_ms": total_time * 1000,
                    "framework_used": self.implementation_type.value,
                    "mcp_server_interactions": len([p for p in processing_pipeline if p["status"] == "completed"]),
                    "processing_session_id": f"session_{int(start_time * 1000)}_{self.implementation_type.value}"
                }
            }
            
            # End processing successfully
            final_result = {
                'status': 'success',
                'total_processing_time': total_time,
                'framework_processing_time': result.processing_time if hasattr(result, 'processing_time') else 0.0,
                'document_type': result.classification.document_type.value if hasattr(result, 'classification') and result.classification else 'unknown',
                'overall_accuracy': result.accuracy_assessment.overall_accuracy if hasattr(result, 'accuracy_assessment') and result.accuracy_assessment else 0.0,
                'confidence': result.classification.confidence if hasattr(result, 'classification') and result.classification else 0.0,
                'implementation': self.implementation_type.value,
                'step_details': step_details,
                'timing_details': timing_details,
                'final_decision_json': final_decision_json,
                'classification_json': classification_json,
                'extraction_json': extraction_json,
                'accuracy_json': accuracy_json,
                'processing_pipeline': processing_pipeline,
                'complete_trace_route': final_decision_json.get('complete_trace_route', {})
            }
            
            # Save enhanced trace file
            trace_data_for_file = {
                'start_time': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
                'end_time': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time + total_time)),
                'total_processing_time_ms': total_time * 1000,
                'total_processing_time': total_time,
                'framework_processing_time': result.processing_time if hasattr(result, 'processing_time') else 0.0,
                'processing_pipeline': processing_pipeline,
                'step_details': step_details,
                'timing_details': timing_details,
                'classification_json': classification_json,
                'extraction_json': extraction_json,
                'accuracy_json': accuracy_json,
                'final_decision_json': final_decision_json,
                'overall_accuracy': result.accuracy_assessment.overall_accuracy if hasattr(result, 'accuracy_assessment') and result.accuracy_assessment else 0.0,
                'confidence': result.classification.confidence if hasattr(result, 'classification') and result.classification else 0.0,
                'document_type': result.classification.document_type.value if hasattr(result, 'classification') and result.classification else 'unknown',
                'status': 'success',
                'success_rate': 1.0
            }
            
            # Save trace file
            self._save_trace_file(trace_data_for_file, document_path)
            
            # Record enhanced result in progress collector
            progress_collector.record_enhanced_result(
                implementation=self.implementation_type,
                document_path=document_path,
                enhanced_result=final_result
            )
            
            progress_collector.end_processing(
                self.implementation_type,
                document_path,
                success=True,
                result=final_result
            )
            
            return final_result
            
        except Exception as e:
            # End processing with error
            error_result = {
                'status': 'error',
                'error': str(e),
                'implementation': self.implementation_type.value
            }
            
            progress_collector.end_processing(
                self.implementation_type,
                document_path,
                success=False,
                result=error_result
            )
            
            return error_result


class DocumentProcessorFactory:
    """Factory for creating document processors with their wrappers"""
    
    @staticmethod
    def build(impl_type: ImplementationType, progress_collector: Optional[ProgressCollectorMCP] = None) -> DocumentProcessorBase:
        """Create a document processor based on implementation type"""
        if impl_type == ImplementationType.CREWAI:
            impl = CrewAIDocumentProcessor()
            processor = DocumentProcessorBase(impl, impl_type)
            if progress_collector is not None:
                processor.progress_collector = progress_collector
            return processor
        
        elif impl_type == ImplementationType.LANGCHAIN:
            impl = LangChainDocumentProcessor()
            processor = DocumentProcessorBase(impl, impl_type)
            if progress_collector is not None:
                processor.progress_collector = progress_collector
            return processor
        
        else:
            raise ValueError(f"Unsupported implementation type: {impl_type}")
    

    
    @staticmethod
    def build_all(progress_collector: ProgressCollectorMCP) -> Dict[str, DocumentProcessorBase]:
        """Create all enhanced processors with progress collector"""
        return {
            'crewai': DocumentProcessorFactory.build(ImplementationType.CREWAI, progress_collector),
            'langchain': DocumentProcessorFactory.build(ImplementationType.LANGCHAIN, progress_collector)
        }


class BatchProcessor:
    """Batch processor for comparing multiple documents across implementations"""
    
    def __init__(self, progress_collector: ProgressCollectorMCP):
        self.progress_collector = progress_collector
        self.crewai_processor = DocumentProcessorFactory.build(ImplementationType.CREWAI, progress_collector)
        self.langchain_processor = DocumentProcessorFactory.build(ImplementationType.LANGCHAIN, progress_collector)
    
    async def compare_implementations(self, document_paths: List[str]) -> Dict[str, Any]:
        """Compare both implementations on the same set of documents"""
        
        print(f"\n🔄 Starting batch comparison of {len(document_paths)} documents...")
        
        # Results storage
        crewai_results = []
        langchain_results = []
        
        # Process with CrewAI
        print(f"\n🤖 Processing with CrewAI...")
        crewai_start_time = time.time()
        
        for i, doc_path in enumerate(document_paths, 1):
            try:
                filename = Path(doc_path).name
                print(f"   📄 {i}/{len(document_paths)}: {filename}")
                
                result = await self.crewai_processor.process(doc_path)
                crewai_results.append({
                    'document_path': doc_path,
                    'result': result,
                    'success': True,
                    'error': None
                })
                
            except Exception as e:
                print(f"      ❌ Error: {str(e)[:50]}...")
                crewai_results.append({
                    'document_path': doc_path,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
        
        crewai_total_time = time.time() - crewai_start_time
        
        # Process with LangChain
        print(f"\n🦜 Processing with LangChain...")
        langchain_start_time = time.time()
        
        for i, doc_path in enumerate(document_paths, 1):
            try:
                filename = Path(doc_path).name
                print(f"   📄 {i}/{len(document_paths)}: {filename}")
                
                result = await self.langchain_processor.process(doc_path)
                langchain_results.append({
                    'document_path': doc_path,
                    'result': result,
                    'success': True,
                    'error': None
                })
                
            except Exception as e:
                print(f"      ❌ Error: {str(e)[:50]}...")
                langchain_results.append({
                    'document_path': doc_path,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
        
        langchain_total_time = time.time() - langchain_start_time
        
        # Calculate comparison metrics
        crewai_successful = len([r for r in crewai_results if r['success']])
        crewai_failed = len([r for r in crewai_results if not r['success']])
        crewai_success_rate = crewai_successful / len(crewai_results) if crewai_results else 0
        
        langchain_successful = len([r for r in langchain_results if r['success']])
        langchain_failed = len([r for r in langchain_results if not r['success']])
        langchain_success_rate = langchain_successful / len(langchain_results) if langchain_results else 0
        
        # Determine winners
        faster_implementation = "CrewAI" if crewai_total_time < langchain_total_time else "LangChain"
        more_reliable = "CrewAI" if crewai_success_rate > langchain_success_rate else "LangChain"
        time_difference = abs(crewai_total_time - langchain_total_time)
        
        comparison_results = {
            'document_count': len(document_paths),
            'crewai': {
                'successful': crewai_successful,
                'failed': crewai_failed,
                'success_rate': crewai_success_rate,
                'total_time': crewai_total_time,
                'avg_time_per_doc': crewai_total_time / len(document_paths) if document_paths else 0,
                'results': crewai_results
            },
            'langchain': {
                'successful': langchain_successful,
                'failed': langchain_failed,
                'success_rate': langchain_success_rate,
                'total_time': langchain_total_time,
                'avg_time_per_doc': langchain_total_time / len(document_paths) if document_paths else 0,
                'results': langchain_results
            },
            'faster_implementation': faster_implementation,
            'more_reliable': more_reliable,
            'time_difference': time_difference
        }
        
        return {
            'comparison': comparison_results,
            'crewai_results': crewai_results,
            'langchain_results': langchain_results
        }
    
    async def process_batch_single_implementation(self, 
                                                implementation: ImplementationType,
                                                document_paths: List[str]) -> Dict[str, Any]:
        """Process a batch of documents with a single implementation"""
        
        if implementation == ImplementationType.CREWAI:
            processor = self.crewai_processor
            impl_name = "CrewAI"
        else:
            processor = self.langchain_processor
            impl_name = "LangChain"
        
        print(f"\n🔄 Processing {len(document_paths)} documents with {impl_name}...")
        
        results = []
        start_time = time.time()
        
        for i, doc_path in enumerate(document_paths, 1):
            try:
                filename = Path(doc_path).name
                print(f"   📄 {i}/{len(document_paths)}: {filename}")
                
                result = await processor.process(doc_path)
                results.append({
                    'document_path': doc_path,
                    'result': result,
                    'success': True,
                    'error': None
                })
                
                # Show basic result info
                if hasattr(result, 'classification'):
                    print(f"      📋 {result.classification.document_type} (conf: {result.classification.confidence:.2f})")
                else:
                    print(f"      ✅ Processed successfully")
                    
            except Exception as e:
                print(f"      ❌ Error: {str(e)[:50]}...")
                results.append({
                    'document_path': doc_path,
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
        
        total_time = time.time() - start_time
        successful = len([r for r in results if r['success']])
        failed = len([r for r in results if not r['success']])
        success_rate = successful / len(results) if results else 0
        
        print(f"\n📊 {impl_name} Batch Results:")
        print(f"   ✅ Successful: {successful}/{len(document_paths)}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📈 Success Rate: {success_rate:.1%}")
        print(f"   ⏱️  Total Time: {total_time:.2f}s")
        print(f"   📊 Avg Time/Doc: {total_time/len(document_paths):.2f}s")
        
        return {
            'implementation': implementation.value,
            'total_documents': len(document_paths),
            'successful': successful,
            'failed': failed,
            'success_rate': success_rate,
            'total_time': total_time,
            'avg_time_per_doc': total_time / len(document_paths) if document_paths else 0,
            'results': results
        }
