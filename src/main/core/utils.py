try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV not available - using mock implementation")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("NumPy not available")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available")

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("Pytesseract not available")

import base64
import io
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Utility class for image processing operations."""
    
    @staticmethod
    def load_image(image_path: str) -> Any:
        """Load image from file path."""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available - returning mock image data")
            return {"mock_image": image_path}
            
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            return image
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise
    
    @staticmethod
    def preprocess_image(image: Any) -> Any:
        """Preprocess image for better OCR results."""
        if not CV2_AVAILABLE or not NUMPY_AVAILABLE:
            logger.warning("OpenCV/NumPy not available - returning original image")
            return image
            
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Resize image if too small (helps with OCR accuracy)
            height, width = gray.shape
            if height < 300 or width < 300:
                scale_factor = max(300 / height, 300 / width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # Apply bilateral filter to reduce noise while keeping edges sharp
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(filtered)
            
            # Apply morphological operations to clean up text
            kernel = np.ones((1,1), np.uint8)
            cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
            
            # Apply adaptive threshold for better text separation
            thresh = cv2.adaptiveThreshold(
                cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            return thresh
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image
    
    @staticmethod
    def image_to_base64(image: Any) -> str:
        """Convert numpy array image to base64 string."""
        if not CV2_AVAILABLE or not PIL_AVAILABLE:
            logger.warning("OpenCV/PIL not available - returning mock base64")
            return "mock_base64_image_data"
            
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Convert to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return img_str
        except Exception as e:
            logger.error(f"Base64 conversion failed: {e}")
            return "error_converting_image"
    
    @staticmethod
    def extract_text_with_ocr(image: Any) -> Dict[str, Any]:
        """Extract text using OCR with bounding box information."""
        if not PYTESSERACT_AVAILABLE or not PIL_AVAILABLE:
            logger.warning("Pytesseract/PIL not available - returning mock OCR data")
            return {
                'text': ['Sample', 'Text', 'From', 'Document'],
                'confidence': [85, 90, 88, 92],
                'bounding_boxes': [
                    {'x': 100, 'y': 50, 'width': 80, 'height': 20},
                    {'x': 200, 'y': 50, 'width': 60, 'height': 20},
                    {'x': 100, 'y': 80, 'width': 70, 'height': 20},
                    {'x': 200, 'y': 80, 'width': 100, 'height': 20}
                ]
            }
            
        try:
            # Try multiple preprocessing approaches for better OCR
            best_result = None
            best_confidence = 0
            
            # Approach 1: Original image with minimal processing
            original_result = ImageProcessor._extract_with_config(image, "--psm 6")
            
            # Approach 2: Preprocessed image
            processed_image = ImageProcessor.preprocess_image(image)
            processed_result = ImageProcessor._extract_with_config(processed_image, "--psm 6")
            
            # Approach 3: Different PSM modes for documents
            psm_modes = ["--psm 4", "--psm 6", "--psm 8", "--psm 11", "--psm 13"]
            
            all_results = [original_result, processed_result]
            
            for psm in psm_modes:
                result = ImageProcessor._extract_with_config(image, psm)
                all_results.append(result)
                
                # Try with preprocessing too
                processed_result_psm = ImageProcessor._extract_with_config(processed_image, psm)
                all_results.append(processed_result_psm)
            
            # Find the result with most text and reasonable confidence
            for result in all_results:
                if result and result['text']:
                    total_text = ' '.join(result['text'])
                    avg_confidence = sum(result['confidence']) / len(result['confidence']) if result['confidence'] else 0
                    text_score = len(total_text) * (avg_confidence / 100)
                    
                    if text_score > best_confidence:
                        best_confidence = text_score
                        best_result = result
                        
            final_result = best_result if best_result else {'text': [], 'confidence': [], 'bounding_boxes': []}
            logger.info("OCR extraction results:\n%s", json.dumps(final_result, indent=2))
            return final_result
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {'text': [], 'confidence': [], 'bounding_boxes': []}
    
    @staticmethod
    def _extract_with_config(image: Any, config: str) -> Dict[str, Any]:
        """Extract text with specific Tesseract configuration."""
        if not PYTESSERACT_AVAILABLE or not PIL_AVAILABLE:
            return {'text': [], 'confidence': [], 'bounding_boxes': []}
            
        try:
            # Convert to PIL Image for pytesseract
            if hasattr(image, 'shape'):  # numpy array
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # BGR to RGB conversion
                    if CV2_AVAILABLE:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        image_rgb = image
                    pil_image = Image.fromarray(image_rgb)
                else:
                    pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Extract text with detailed information using custom config
            custom_config = f'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,/:-() {config}'
            data = pytesseract.image_to_data(pil_image, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Filter out low confidence results
            filtered_data = {
                'text': [],
                'confidence': [],
                'bounding_boxes': []
            }
            
            for i, conf in enumerate(data['conf']):
                if conf > 10:  # Lower threshold to capture more text
                    text = data['text'][i].strip()
                    if text and len(text) > 1:  # Ignore single characters unless they're numbers
                        filtered_data['text'].append(text)
                        filtered_data['confidence'].append(conf)
                        filtered_data['bounding_boxes'].append({
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        })
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"OCR extraction with config {config} failed: {e}")
            return {'text': [], 'confidence': [], 'bounding_boxes': []}

class DocumentTemplates:
    """Templates for different document types with expected fields."""
    
    ID_CARD_FIELDS = [
        "name", "surname", "date_of_birth", "id_number", 
        "address", "nationality", "gender", "issue_date", "expiry_date"
    ]
    
    DRIVER_LICENSE_FIELDS = [
        "name", "surname", "date_of_birth", "license_number", 
        "address", "class", "issue_date", "expiry_date", "restrictions"
    ]
    
    PASSPORT_FIELDS = [
        "name", "surname", "passport_number", "date_of_birth", 
        "place_of_birth", "nationality", "gender", "issue_date", "expiry_date"
    ]
    
    @classmethod
    def get_expected_fields(cls, document_type: str) -> List[str]:
        """Get expected fields for a document type."""
        mapping = {
            "id_card": cls.ID_CARD_FIELDS,
            "driver_license": cls.DRIVER_LICENSE_FIELDS,
            "passport": cls.PASSPORT_FIELDS
        }
        return mapping.get(document_type, [])

class MCPClient:
    """Simple MCP client for external service integration."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    async def call_service(self, method: str, params: Dict) -> Dict:
        """Call MCP service method."""
        # This is a placeholder - implement actual MCP protocol
        # For now, return mock data
        return {
            "result": f"Mock MCP response for {method}",
            "params": params
        }
