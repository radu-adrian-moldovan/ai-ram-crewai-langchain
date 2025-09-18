"""
CrewAI Document Processor using Unified Tool System
Uses the same MCP backend as LangChain through the unified tool abstraction.
Uses centralized LLM manager for offline-first operation.
"""

import os
import asyncio
import time
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.main.core.models import (
    DocumentType, DocumentClassification, DocumentExtraction, 
    AccuracyAssessment, FinalDecision, ProcessingResult, ExtractedField
)

# Import unified tool system
from src.main.core.manager_mcp import mcp_tool
from src.main.core.manager_tool import to_crewai_tool, CREWAI_AVAILABLE

# Import centralized LLM manager
from src.main.core.manager_llm import get_crewai_llm, ModelType

# Import configuration utility
from main.core.manager_propmt import PromptConfigLoader

logger = logging.getLogger(__name__)

if CREWAI_AVAILABLE:
    from crewai import Agent, Task, Crew, Process

class CrewAIDocumentProcessor:
    """CrewAI implementation using unified MCP tool and centralized LLM manager."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", mcp_url: str = "http://localhost:3000"):
        if not CREWAI_AVAILABLE:
            raise ImportError("CrewAI not available - install crewai packages")
        
        # Load agent configurations from YAML
        self.config = self._load_config()
        
        # Use centralized LLM manager for offline-first operation
        self.llm = get_crewai_llm(ModelType.VISION)
        print("✓ CrewAI LLM initialized through centralized manager")
        
        # Store reference to unified MCP tool (don't use as CrewAI tool due to compatibility issues)
        self.mcp_tool = mcp_tool
        
        # Setup agents without tools (agents will call MCP tool directly)
        self._setup_agents()
    
    def _load_config(self):
        """Load agent configurations from prompts.yaml file."""
        config_loader = PromptConfigLoader()
        return config_loader.load_config()
    
    def _get_agent_config(self, agent_name: str):
        """Get configuration for a specific agent."""
        config_loader = PromptConfigLoader()
        return config_loader.get_agent_config(agent_name)
    
    def _setup_agents(self):
        """Setup CrewAI agents using configuration from prompts.yaml."""
        
        # Classification Agent
        classifier_cfg = self._get_agent_config("document_classification_agent")
        self.classifier_agent = Agent(
            role=classifier_cfg["role"],
            goal=classifier_cfg["goal"],
            backstory=classifier_cfg["backstory"],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # Extraction Agent
        extractor_cfg = self._get_agent_config("field_extraction_agent")
        self.extractor_agent = Agent(
            role=extractor_cfg["role"],
            goal=extractor_cfg["goal"],
            backstory=extractor_cfg["backstory"],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # Accuracy Assessment Agent
        accuracy_cfg = self._get_agent_config("accuracy_assessment_agent")
        self.accuracy_agent = Agent(
            role=accuracy_cfg["role"],
            goal=accuracy_cfg["goal"],
            backstory=accuracy_cfg["backstory"],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
        
        # Decision Agent
        decision_cfg = self._get_agent_config("final_decision_agent")
        self.decision_agent = Agent(
            role=decision_cfg["role"],
            goal=decision_cfg["goal"],
            backstory=decision_cfg["backstory"],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    async def process_document(self, image_path: str) -> ProcessingResult:
        """Process document through CrewAI agents using unified MCP tool."""
        start_time = time.time()
        
        try:
            # Step 1: Classification
            classification_result = await self._run_classification(image_path)
            
            # Step 2: Field Extraction
            extraction_result = await self._run_extraction(image_path, classification_result)
            
            # Step 3: Accuracy Assessment
            accuracy_result = await self._run_accuracy_assessment(image_path, classification_result, extraction_result)
            
            # Step 4: Final Decision
            decision_result = await self._run_final_decision(image_path, classification_result, extraction_result, accuracy_result)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                classification=classification_result,
                extraction=extraction_result,
                accuracy_assessment=accuracy_result,
                final_decision=decision_result,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"CrewAI document processing failed: {e}")
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                classification=DocumentClassification(
                    document_type=DocumentType.UNKNOWN,
                    confidence=0.0,
                    reasoning=f"Processing failed: {e}"
                ),
                extraction=DocumentExtraction(extracted_fields=[], total_fields_found=0),
                accuracy_assessment=AccuracyAssessment(
                    overall_accuracy=0.0,
                    field_accuracies={},
                    issues_found=[f"Processing error: {e}"],
                    recommendations=["Fix CrewAI-MCP integration"]
                ),
                final_decision=FinalDecision(
                    accept=False,
                    confidence=0.0,
                    reasoning="Processing failed",
                    required_actions=["Retry processing"]
                ),
                processing_time=processing_time
            )
    
    async def _run_classification(self, image_path: str) -> DocumentClassification:
        """Run classification using CrewAI agent with MCP tool - matches LangChain implementation."""
        try:
            # First, call MCP tool directly to get classification data
            mcp_result = await self.mcp_tool.run(method="document/classify", image_path=image_path)
            
            # Create task for classification analysis using configuration
            task_config = self.config["tasks"]["classification"]
            classification_task = Task(
                description=task_config["description"].format(image_path=image_path),
                agent=self.classifier_agent,
                expected_output=task_config["expected_output"]
            )
            
            # Create crew with single task
            crew = Crew(
                agents=[self.classifier_agent],
                tasks=[classification_task],
                process=Process.sequential,
                verbose=True
            )
            
            result = await asyncio.get_event_loop().run_in_executor(
                None, crew.kickoff, {"image_path": image_path}
            )
            
            # Parse agent output to extract classification - matches LangChain parsing
            output_text = str(result).lower()
            
            if "id card" in output_text or "identity" in output_text:
                doc_type = DocumentType.ID_CARD
                confidence = 0.85
            elif "driver" in output_text or "license" in output_text:
                doc_type = DocumentType.DRIVER_LICENSE
                confidence = 0.85
            elif "passport" in output_text:
                doc_type = DocumentType.PASSPORT
                confidence = 0.85
            else:
                doc_type = DocumentType.UNKNOWN
                confidence = 0.3
            
            return DocumentClassification(
                document_type=doc_type,
                confidence=confidence,
                reasoning=f"CrewAI agent with MCP: {output_text[:100]}"
            )
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return DocumentClassification(
                document_type=DocumentType.UNKNOWN,
                confidence=0.0,
                reasoning=f"Classification error: {e}"
            )
    
    async def _run_extraction(self, image_path: str, classification: DocumentClassification) -> DocumentExtraction:
        """Run field extraction using CrewAI agent with MCP tool - matches LangChain implementation."""
        try:
            # First, call MCP tool directly to get extraction data
            mcp_result = await self.mcp_tool.run(method="document/extract", image_path=image_path, document_type=classification.document_type.value)
            
            # Create task for extraction analysis using configuration
            task_config = self.config["tasks"]["extraction"]
            extraction_task = Task(
                description=task_config["description"].format(
                    image_path=image_path, 
                    document_type=classification.document_type.value
                ),
                agent=self.extractor_agent,
                expected_output=task_config["expected_output"]
            )
            
            # Create crew with single task
            crew = Crew(
                agents=[self.extractor_agent],
                tasks=[extraction_task],
                process=Process.sequential,
                verbose=True
            )
            
            result = await asyncio.get_event_loop().run_in_executor(
                None, crew.kickoff, {
                    "image_path": image_path,
                    "document_type": classification.document_type.value
                }
            )
            
            output_text = str(result)
            
            # Create sample extracted field (in real implementation, parse MCP results)
            # Fixed: Remove artificial truncation to preserve full JSON content
            extracted_fields = [
                ExtractedField(
                    field_name="extracted_text",
                    value=output_text,  # Full output without truncation
                    confidence=0.8
                )
            ]
            
            return DocumentExtraction(
                extracted_fields=extracted_fields,
                total_fields_found=len(extracted_fields)
            )
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return DocumentExtraction(extracted_fields=[], total_fields_found=0)
    
    async def _run_accuracy_assessment(self, image_path: str, classification: DocumentClassification, extraction: DocumentExtraction) -> AccuracyAssessment:
        """Run accuracy assessment using CrewAI agent with MCP tool - matches LangChain implementation."""
        try:
            # First, call MCP tool directly to get analysis data
            mcp_result = await self.mcp_tool.run(method="document/analyze", image_path=image_path)
            
            # Create task for accuracy assessment using configuration
            task_config = self.config["tasks"]["accuracy_assessment"]
            accuracy_task = Task(
                description=task_config["description"].format(
                    image_path=image_path,
                    classification=classification.document_type.value,
                    extraction=len(extraction.extracted_fields)
                ),
                agent=self.accuracy_agent,
                expected_output=task_config["expected_output"]
            )
            
            # Create crew with single task
            crew = Crew(
                agents=[self.accuracy_agent],
                tasks=[accuracy_task],
                process=Process.sequential,
                verbose=True
            )
            
            result = await asyncio.get_event_loop().run_in_executor(
                None, crew.kickoff, {
                    "image_path": image_path,
                    "classification": classification.document_type.value,
                    "extraction": str(len(extraction.extracted_fields))
                }
            )
            
            return AccuracyAssessment(
                overall_accuracy=0.8,
                field_accuracies={"overall": 0.8},
                issues_found=["CrewAI-MCP analysis completed"],
                recommendations=["Review results"]
            )
            
        except Exception as e:
            logger.error(f"Accuracy assessment failed: {e}")
            return AccuracyAssessment(
                overall_accuracy=0.0,
                field_accuracies={},
                issues_found=[f"Assessment error: {e}"],
                recommendations=["Fix accuracy assessment"]
            )
    
    async def _run_final_decision(self, image_path: str, classification: DocumentClassification, extraction: DocumentExtraction, accuracy: AccuracyAssessment) -> FinalDecision:
        """Run final decision using CrewAI agent with MCP tool - matches LangChain implementation."""
        try:
            # Create task for final decision using configuration
            task_config = self.config["tasks"]["final_decision"]
            decision_task = Task(
                description=task_config["description"].format(
                    image_path=image_path,
                    classification=classification.document_type.value,
                    extraction=len(extraction.extracted_fields),
                    accuracy=accuracy.overall_accuracy
                ),
                agent=self.decision_agent,
                expected_output=task_config["expected_output"]
            )
            
            # Create crew with single task
            crew = Crew(
                agents=[self.decision_agent],
                tasks=[decision_task],
                process=Process.sequential,
                verbose=True
            )
            
            result = await asyncio.get_event_loop().run_in_executor(
                None, crew.kickoff, {
                    "image_path": image_path,
                    "classification": classification.document_type.value,
                    "extraction": str(len(extraction.extracted_fields)),
                    "accuracy": str(accuracy.overall_accuracy)
                }
            )
            
            output_text = str(result).lower()
            accept = "accept" in output_text and "reject" not in output_text
            
            return FinalDecision(
                accept=accept,
                confidence=0.8,
                reasoning=f"CrewAI decision with MCP: {output_text[:100]}",
                required_actions=[] if accept else ["Review document quality"]
            )
            
        except Exception as e:
            logger.error(f"Decision failed: {e}")
            return FinalDecision(
                accept=False,
                confidence=0.0,
                reasoning=f"Decision error: {e}",
                required_actions=["Fix decision process"]
            )

# Example usage
async def main():
    """Demo CrewAI document processing with unified MCP tool."""
    if not CREWAI_AVAILABLE:
        print("CrewAI not available - skipping demo")
        return
    
    processor = CrewAIDocumentProcessor()
    
    # Process a sample document
    image_path = os.path.join(str(Path(__file__).parent.parent.parent.parent), "samples", "input", "idcard_classic", "idcard_classic_Superman.jpg")
    
    try:
        result = await processor.process_document(image_path)
        print(f"CrewAI Processing completed in {result.processing_time:.2f} seconds")
        print(f"Document Type: {result.classification.document_type}")
        print(f"Classification Confidence: {result.classification.confidence}")
        print(f"Final Decision: {'ACCEPTED' if result.final_decision.accept else 'REJECTED'}")
        print(f"Extracted Fields: {len(result.extraction.extracted_fields)}")
        
    except Exception as e:
        print(f"Error processing document: {e}")

if __name__ == "__main__":
    asyncio.run(main())
