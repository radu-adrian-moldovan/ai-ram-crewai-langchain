from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from enum import Enum

class DocumentType(str, Enum):
    """Supported document types for classification."""
    ID_CARD = "id_card"
    DRIVER_LICENSE = "driver_license"
    PASSPORT = "passport"
    UNKNOWN = "unknown"

class ExtractedField(BaseModel):
    """Represents an extracted field from a document."""
    field_name: str = Field(description="Name of the extracted field")
    value: str = Field(description="Extracted value")
    confidence: float = Field(description="Confidence score between 0 and 1")
    bounding_box: Optional[Dict[str, int]] = Field(default=None, description="Bounding box coordinates")

class DocumentClassification(BaseModel):
    """Result of document classification."""
    document_type: DocumentType = Field(description="Classified document type")
    confidence: float = Field(description="Classification confidence score")
    reasoning: str = Field(description="Explanation for the classification")

class DocumentExtraction(BaseModel):
    """Result of field extraction from document."""
    extracted_fields: List[ExtractedField] = Field(description="List of extracted fields")
    total_fields_found: int = Field(description="Total number of fields extracted")
    
class AccuracyAssessment(BaseModel):
    """Assessment of extraction accuracy."""
    overall_accuracy: float = Field(description="Overall accuracy score")
    field_accuracies: Dict[str, float] = Field(description="Individual field accuracy scores")
    issues_found: List[str] = Field(description="List of issues or concerns")
    recommendations: List[str] = Field(description="Recommendations for improvement")

class FinalDecision(BaseModel):
    """Final decision on document processing."""
    accept: bool = Field(description="Whether to accept the document")
    confidence: float = Field(description="Confidence in the decision")
    reasoning: str = Field(description="Explanation for the decision")
    required_actions: List[str] = Field(description="Actions required if document is not accepted")

class ProcessingResult(BaseModel):
    """Complete processing result."""
    classification: DocumentClassification
    extraction: DocumentExtraction
    accuracy_assessment: AccuracyAssessment
    final_decision: FinalDecision
    processing_time: float = Field(description="Total processing time in seconds")
