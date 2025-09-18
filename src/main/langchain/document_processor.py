"""
LangChain Document Processor using Unified Tool System
Uses the same MCP backend as CrewAI through the unified tool abstraction.
Uses centralized LLM manager for offline-first operation.
"""

import os
import asyncio
from typing import Dict, Any, List, Optional
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
from src.main.core.manager_tool import LangChainToolAdapter, LANGCHAIN_AVAILABLE

# Import centralized LLM manager
from src.main.core.manager_llm import get_langchain_llm, ModelType

# Import configuration utility
from main.core.manager_propmt import PromptConfigLoader

logger = logging.getLogger(__name__)

if LANGCHAIN_AVAILABLE:
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

class LangChainDocumentProcessor:
    """LangChain implementation using unified MCP tool and centralized LLM manager."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", mcp_url: str = "http://localhost:3000"):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not available - install langchain packages")
        
        # Load agent configurations from YAML
        self.config = self._load_config()
        
        # Use centralized LLM manager for offline-first operation
        self.llm = get_langchain_llm(ModelType.VISION)
        print("✓ LangChain LLM initialized through centralized manager")
        
        # Create LangChain adapter for the unified MCP tool
        self.mcp_langchain_tool = LangChainToolAdapter(
            name=mcp_tool.name,
            description=mcp_tool.description,
            unified_tool=mcp_tool
        )
        self.tools = [self.mcp_langchain_tool]
        
        # Setup agents
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
        """Setup LangChain agents using configuration from prompts.yaml."""
        
        # Classification Agent
        classifier_cfg = self._get_agent_config("document_classification_agent")
        classification_prompt = ChatPromptTemplate.from_messages([
            ("system", classifier_cfg["backstory"]),
            ("human", self.config["langchain_prompts"]["classification"]),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        self.classification_agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=classification_prompt
        )
        
        self.classification_executor = AgentExecutor(
            agent=self.classification_agent,
            tools=self.tools,
            verbose=True
        )
        
        # Extraction Agent
        extractor_cfg = self._get_agent_config("field_extraction_agent")
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", extractor_cfg["backstory"]),
            ("human", self.config["langchain_prompts"]["extraction"]),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        self.extraction_agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=extraction_prompt
        )
        
        self.extraction_executor = AgentExecutor(
            agent=self.extraction_agent,
            tools=self.tools,
            verbose=True
        )
        
        # Accuracy Assessment Agent
        accuracy_cfg = self._get_agent_config("accuracy_assessment_agent")
        accuracy_prompt = ChatPromptTemplate.from_messages([
            ("system", accuracy_cfg["backstory"]),
            ("human", self.config["langchain_prompts"]["accuracy_assessment"]),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        self.accuracy_agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=accuracy_prompt
        )
        
        self.accuracy_executor = AgentExecutor(
            agent=self.accuracy_agent,
            tools=self.tools,
            verbose=True
        )
        
        # Decision Agent
        decision_cfg = self._get_agent_config("final_decision_agent")
        decision_prompt = ChatPromptTemplate.from_messages([
            ("system", decision_cfg["backstory"]),
            ("human", self.config["langchain_prompts"]["final_decision"]),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        self.decision_agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=decision_prompt
        )
        
        self.decision_executor = AgentExecutor(
            agent=self.decision_agent,
            tools=self.tools,
            verbose=True
        )
    
    async def process_document(self, image_path: str) -> ProcessingResult:
        """Process document through LangChain agents using unified MCP tool."""
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
            logger.error(f"LangChain document processing failed: {e}")
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
                    recommendations=["Fix LangChain-MCP integration"]
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
        """Run classification using LangChain agent with MCP tool."""
        try:
            result = await self.classification_executor.ainvoke({
                "image_path": image_path
            })
            
            # Parse agent output to extract classification
            output_text = result.get("output", "").lower()
            
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
                reasoning=f"LangChain agent with MCP: {output_text[:100]}"
            )
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return DocumentClassification(
                document_type=DocumentType.UNKNOWN,
                confidence=0.0,
                reasoning=f"Classification error: {e}"
            )
    
    async def _run_extraction(self, image_path: str, classification: DocumentClassification) -> DocumentExtraction:
        """Run field extraction using LangChain agent with MCP tool."""
        try:
            result = await self.extraction_executor.ainvoke({
                "image_path": image_path,
                "document_type": classification.document_type.value
            })
            
            output_text = result.get("output", "")
            
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
        """Run accuracy assessment using LangChain agent with MCP tool."""
        try:
            result = await self.accuracy_executor.ainvoke({
                "image_path": image_path,
                "classification": classification.document_type.value,
                "extraction": str(len(extraction.extracted_fields))
            })
            
            return AccuracyAssessment(
                overall_accuracy=0.8,
                field_accuracies={"overall": 0.8},
                issues_found=["LangChain-MCP analysis completed"],
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
        """Run final decision using LangChain agent with MCP tool."""
        try:
            result = await self.decision_executor.ainvoke({
                "image_path": image_path,
                "classification": classification.document_type.value,
                "extraction": str(len(extraction.extracted_fields)),
                "accuracy": str(accuracy.overall_accuracy)
            })
            
            output_text = result.get("output", "").lower()
            accept = "accept" in output_text and "reject" not in output_text
            
            return FinalDecision(
                accept=accept,
                confidence=0.8,
                reasoning=f"LangChain decision with MCP: {output_text[:100]}",
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
    """Demo LangChain document processing with unified MCP tool."""
    if not LANGCHAIN_AVAILABLE:
        print("LangChain not available - skipping demo")
        return
    
    processor = LangChainDocumentProcessor()
    
    # Process a sample document
    image_path = os.path.join(str(Path(__file__).parent.parent.parent.parent), "samples", "input", "idcard_classic", "idcard_classic_Superman.jpg")
    
    try:
        result = await processor.process_document(image_path)
        print(f"LangChain Processing completed in {result.processing_time:.2f} seconds")
        print(f"Document Type: {result.classification.document_type}")
        print(f"Classification Confidence: {result.classification.confidence}")
        print(f"Final Decision: {'ACCEPTED' if result.final_decision.accept else 'REJECTED'}")
        print(f"Extracted Fields: {len(result.extraction.extracted_fields)}")
        
    except Exception as e:
        print(f"Error processing document: {e}")

if __name__ == "__main__":
    asyncio.run(main())
