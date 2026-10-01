import cv2
import numpy as np
import pytesseract
import logging
import re
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Fallback path for Windows if Tesseract is installed in the default location
# but not added to PATH. 
try:
    pytesseract.get_tesseract_version()
except pytesseract.TesseractNotFoundError:
    tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        logger.warning(f"Tesseract OCR not found. OCR features will be disabled. {e}")

def preprocess_for_ocr(image_path: str) -> np.ndarray:
    """Preprocess image to improve OCR accuracy."""
    img = cv2.imread(image_path)
    if img is None:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Scale up if image is too small (helps tesseract read small text)
    h, w = gray.shape
    if h < 500 or w < 500:
        gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        
    # Apply slight blur to remove noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def extract_text(image_path: str) -> Dict[str, Any]:
    """
    Extracts text from an image using Tesseract OCR.
    Returns the extracted string and a confidence score.
    """
    try:
        processed_img = preprocess_for_ocr(image_path)
        if processed_img is None:
            return {"text": None, "confidence": 0.0}
            
        # Get verbose data including confidence
        data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT)
        
        extracted_text = []
        confidences = []
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = data['conf'][i]
            
            # Filter out empty text and -1 confidences
            if int(conf) > -1 and len(text) > 0:
                extracted_text.append(text)
                confidences.append(float(conf))
                
        # Reconstruct full text
        full_text = " ".join(extracted_text)
        
        # Clean up weird OCR artifacts (multiple spaces, weird punctuation clusters)
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        # Calculate average confidence
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Map Tesseract confidence (0-100) to 0.0-1.0
        normalized_conf = avg_conf / 100.0
        
        if not full_text or len(full_text) < 10:
            return {"text": None, "confidence": 0.0}
            
        return {
            "text": full_text,
            "confidence": normalized_conf
        }
        
    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract not installed. Cannot perform OCR.")
        return {"text": None, "confidence": 0.0}
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return {"text": None, "confidence": 0.0}
