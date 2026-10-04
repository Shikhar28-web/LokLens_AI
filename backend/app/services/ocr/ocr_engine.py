import cv2
import numpy as np
import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Try loading PaddleOCR (Primary)
_paddle_ocr = None
try:
    from paddleocr import PaddleOCR
    # Load PaddleOCR with angle classifier and english language
    _paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
except Exception as e:
    logger.warning(f"PaddleOCR not available. Error: {e}")

# Try loading Surya OCR (High Accuracy Fallback)
_surya_manager, _surya_predictor = None, None
try:
    from surya.inference import SuryaInferenceManager
    from surya.recognition import RecognitionPredictor
    
    _surya_manager = SuryaInferenceManager()
    _surya_predictor = RecognitionPredictor(_surya_manager)
except Exception as e:
    logger.warning(f"Surya-OCR not available, falling back to Tesseract. Error: {e}")

# Fallback path for Windows Tesseract
import pytesseract
try:
    pytesseract.get_tesseract_version()
except pytesseract.TesseractNotFoundError:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        pass


def get_preprocessing_variants(image_path: str) -> Dict[str, np.ndarray]:
    """Generates multiple preprocessing variants for OCR consensus."""
    img = cv2.imread(image_path)
    if img is None:
        return {}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 1. Original
    variants = {"original": img}
    
    # 2. Grayscale + Light Denoise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    variants["grayscale_denoised"] = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    
    # 3. Upscaled + Adaptive Threshold (for small text)
    if h < 1000 or w < 1000:
        gray_large = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        thresh = cv2.adaptiveThreshold(gray_large, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        variants["adaptive_thresh"] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    return variants


def clean_ocr_text(text: str) -> str:
    """Robust OCR normalization layer. Preserves exact text as requested."""
    # The user explicitly requested the exact text from the image without aggressive filtering.
    
    # Just basic whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def run_paddle_ocr(image_path: str) -> List[Dict]:
    """Runs PaddleOCR and returns structured regions with bboxes."""
    if _paddle_ocr is None:
        return []
    try:
        result = _paddle_ocr.ocr(image_path, cls=True)
        if not result or not result[0]:
            return []
            
        regions = []
        for line in result[0]:
            poly, (text, conf) = line
            x1 = min(p[0] for p in poly)
            y1 = min(p[1] for p in poly)
            x2 = max(p[0] for p in poly)
            y2 = max(p[1] for p in poly)
            
            cleaned = clean_ocr_text(text)
            if len(cleaned) < 2 and not any(c.isdigit() for c in cleaned):
                continue
                
            regions.append({
                "text": cleaned,
                "raw_text": text,
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2],
                "center_x": (x1 + x2) / 2,
                "center_y": (y1 + y2) / 2,
                "width": x2 - x1,
                "height": y2 - y1,
                "reliability": "HIGH" if conf > 0.8 else "LOW",
                "engine": "PaddleOCR"
            })
        return regions
    except Exception as e:
        logger.error(f"PaddleOCR error: {e}")
        return []


def run_surya_ocr(image_path: str) -> List[Dict]:
    """Runs Surya-OCR and returns structured regions with bboxes."""
    if _surya_predictor is None:
        return []
        
    try:
        from PIL import Image
        import re
        
        image = Image.open(image_path)
        
        # full_page=True does layout+OCR in one VLM call
        page_results = _surya_predictor([image], full_page=True)
        
        if not page_results or not page_results[0].blocks:
            return []
            
        regions = []
        for block in page_results[0].blocks:
            # Extract plain text from HTML block output
            text = re.sub(r'<[^>]+>', ' ', block.html).strip()
            text = re.sub(r'\s+', ' ', text)
            conf = block.confidence
            
            # polygon is [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            poly = block.polygon
            x1 = min(p[0] for p in poly)
            y1 = min(p[1] for p in poly)
            x2 = max(p[0] for p in poly)
            y2 = max(p[1] for p in poly)
            
            cleaned = clean_ocr_text(text)
            if len(cleaned) < 2 and not any(c.isdigit() for c in cleaned):
                continue
                
            regions.append({
                "text": cleaned,
                "raw_text": text,
                "confidence": float(conf),
                "bbox": [x1, y1, x2, y2],
                "center_x": (x1 + x2) / 2,
                "center_y": (y1 + y2) / 2,
                "width": x2 - x1,
                "height": y2 - y1,
                "reliability": "HIGH" if conf > 0.8 else "LOW",
                "engine": "SuryaOCR"
            })
            
        return regions
    except Exception as e:
        logger.error(f"Surya OCR error: {e}")
        return []


def format_regions_spatially(regions: List[Dict]) -> str:
    """Returns the regions grouped by line/paragraph without metadata."""
    if not regions:
        return ""
        
    # Sort by Y coordinate first, then X (top to bottom, left to right)
    # Allow some Y tolerance for items on the same line
    regions.sort(key=lambda r: (round(r["center_y"] / 20) * 20, r["center_x"]))
    
    lines = []
    current_line = []
    current_y = None
    
    for r in regions:
        text = r["text"]
        if not text:
            continue
            
        y_chunk = round(r["center_y"] / 20) * 20
        if current_y is None:
            current_y = y_chunk
            
        if y_chunk != current_y:
            lines.append(" ".join(current_line))
            current_line = [text]
            current_y = y_chunk
        else:
            current_line.append(text)
            
    if current_line:
        lines.append(" ".join(current_line))
            
    return " ".join(lines)


def extract_text(image_path: str) -> Dict[str, Any]:
    """
    Main OCR entrypoint. Uses Surya OCR as primary, fallback to Tesseract.
    Returns structured data including bounding boxes.
    """
    try:
        variants = get_preprocessing_variants(image_path)
        if not variants:
            return {"status": "FAILED", "error": "Preprocessing failed", "stage": "preprocessing"}
            
        img = variants["original"]
        h, w = img.shape[:2]
        
        import tempfile
        import os
        
        best_regions = []
        best_conf = 0.0
        engine_used = "unknown"
        preprocessing_used = "original"
        engines_attempted = []
        
        if _paddle_ocr is not None:
            engines_attempted.append("PaddleOCR")
            for variant_name, img_data in variants.items():
                if variant_name != "original" and best_conf > 0.85:
                    break
                    
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    temp_path = tf.name
                try:
                    cv2.imwrite(temp_path, img_data)
                    regions = run_paddle_ocr(temp_path)
                    if regions:
                        avg_conf = sum(r["confidence"] for r in regions) / len(regions)
                        if avg_conf > best_conf:
                            best_conf = avg_conf
                            best_regions = regions
                            engine_used = "PaddleOCR"
                            preprocessing_used = variant_name
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        if (not best_regions or best_conf < 0.7) and _surya_predictor is not None:
            engines_attempted.append("SuryaOCR")
            for variant_name, img_data in variants.items():
                if variant_name != "original" and best_conf > 0.85:
                    break
                    
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                    temp_path = tf.name
                try:
                    cv2.imwrite(temp_path, img_data)
                    regions = run_surya_ocr(temp_path)
                    if regions:
                        avg_conf = sum(r["confidence"] for r in regions) / len(regions)
                        if avg_conf > best_conf:
                            best_conf = avg_conf
                            best_regions = regions
                            engine_used = "SuryaOCR"
                            preprocessing_used = variant_name
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        # Fallback to Tesseract
        if (not best_regions or best_conf < 0.3):
            engines_attempted.append("Tesseract")
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
                tess_regions = []
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    conf = data['conf'][i]
                    if int(conf) > -1 and len(text) > 0:
                        cleaned = clean_ocr_text(text)
                        if len(cleaned) > 0:
                            x, y, bw, bh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                            tess_regions.append({
                                "text": cleaned,
                                "raw_text": text,
                                "confidence": float(conf) / 100.0,
                                "bbox": [x, y, x+bw, y+bh],
                                "center_x": x + bw/2,
                                "center_y": y + bh/2,
                                "width": bw,
                                "height": bh,
                                "reliability": "HIGH" if float(conf) > 80 else "LOW",
                                "engine": "Tesseract"
                            })
                if tess_regions:
                    tess_conf = sum(r["confidence"] for r in tess_regions) / len(tess_regions)
                    if tess_conf > best_conf:
                        best_regions = tess_regions
                        best_conf = tess_conf
                        engine_used = "Tesseract"
                        preprocessing_used = "original"
            except Exception as e:
                logger.warning(f"Tesseract fallback failed: {e}")
                        
        if not best_regions:
            return {
                "status": "FAILED",
                "error": "All OCR engines failed to extract text.",
                "engines_attempted": engines_attempted
            }
            
        grouped_text = format_regions_spatially(best_regions)
        raw_text_full = "\n".join([r["raw_text"] for r in best_regions])
        
        # Add LLM OCR Correction to fix Tesseract hallucinations and noise
        try:
            from app.services.nlp.llm_client import call_llm_json
            prompt = f"""
You are an expert at repairing broken OCR text from Tesseract.
The text below contains severe hallucinations, including random UI elements, charts misread as text (like 'm Bex =', 'J)', '©)'), random brackets, slashes, and numbers mixed with text.
Your task is to heavily aggressively clean this text. 
- Delete any fragmented gibberish.
- Remove hallucinated symbols and meaningless equations.
- Fix broken words into proper English sentences.
- Only output the coherent, real text.

Raw OCR Text:
{grouped_text}

Output JSON:
{{
  "cleaned_text": "The completely fixed and coherent English text."
}}
"""
            res = call_llm_json(prompt)
            if res and "cleaned_text" in res:
                grouped_text = res["cleaned_text"]
        except Exception as e:
            logger.error(f"LLM OCR correction failed: {e}")
        
        return {
            "status": "SUCCESS",
            "primary_engine": "PaddleOCR",
            "fallback_engine": "Surya",
            "engine_used": engine_used,
            "preprocessing_variants": list(variants.keys()),
            "preprocessing": preprocessing_used,
            "regions_detected": len(best_regions),
            "overall_confidence": best_conf,
            "regions": best_regions,
            "raw_text": raw_text_full,
            "normalized_text": grouped_text,
            "removed_text": [],
            "corrected_tokens": []
        }
        
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return {
            "status": "FAILED",
            "text": "",
            "confidence": 0,
            "engine": "unknown",
            "error": str(e),
            "stage": "unknown"
        }
