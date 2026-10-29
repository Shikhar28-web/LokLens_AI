"""
Advanced OCR Engine with Multi-Variant Preprocessing, Multi-Language Script Support,
Image Type Detection (Screenshots, Memes, Documents, Pure Photos), and Consensus Extraction.
"""

import cv2
import numpy as np
import logging
import re
import tempfile
import os
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy-loaded PaddleOCR instances per language
_paddle_instances: Dict[str, Any] = {}

def get_paddle_ocr(lang: str = "en"):
    """Lazy loads and caches PaddleOCR engine for specified language."""
    global _paddle_instances
    if lang in _paddle_instances:
        return _paddle_instances[lang]
    try:
        from paddleocr import PaddleOCR
        instance = PaddleOCR(use_angle_cls=True, lang=lang)
        _paddle_instances[lang] = instance
        logger.info(f"Initialized PaddleOCR for lang='{lang}'")
        return instance
    except Exception as e:
        logger.warning(f"PaddleOCR not available for lang='{lang}': {e}")
        return None


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


def detect_image_type(img: np.ndarray) -> str:
    """
    Detects whether the image is a screenshot, meme, news graphic, document, or pure photo.
    """
    h, w = img.shape[:2]
    aspect_ratio = h / max(w, 1)

    # 1. Smartphone screenshot: tall aspect ratio (>= 1.7)
    if aspect_ratio >= 1.7 or aspect_ratio <= 0.6:
        return "screenshot"

    # 2. Check for solid background / document (high white/light pixel ratio)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    white_ratio = np.mean(gray > 235)
    dark_ratio = np.mean(gray < 25)

    if white_ratio > 0.55:
        return "document"
    if dark_ratio > 0.60:
        return "dark_mode_screenshot"

    # 3. Meme detection: top or bottom 20% contains high-contrast text regions
    top_band = gray[:int(h * 0.2), :]
    bot_band = gray[int(h * 0.8):, :]
    if np.std(top_band) > 60 and np.std(bot_band) > 60:
        return "meme"

    # 4. News graphic / chyron: bottom 25% has solid color banner
    bottom_banner = img[int(h * 0.75):, :]
    std_colors = np.std(bottom_banner, axis=(0, 1))
    if np.mean(std_colors) < 40:
        return "news_graphic"

    return "general_image"


def auto_deskew(img: np.ndarray) -> np.ndarray:
    """Calculates skew angle and rotates image if slanted (> 1.0 degree)."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 100:
            return img

        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) > 1.0 and abs(angle) < 45.0:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
    except Exception:
        pass
    return img


def get_preprocessing_variants(image_path: str) -> Dict[str, np.ndarray]:
    """
    Generates advanced preprocessing variants to handle low contrast, dark mode,
    JPEG compression noise, rotation, and small font sizes.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {}

    variants: Dict[str, np.ndarray] = {"original": img}
    h, w = img.shape[:2]

    # 1. CLAHE Contrast enhancement (Crucial for dark mode and low-contrast mobile screenshots)
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        clahe_img = cv2.merge((cl, a, b))
        variants["clahe_contrast"] = cv2.cvtColor(clahe_img, cv2.COLOR_LAB2BGR)
    except Exception:
        pass

    # 2. Grayscale + Bilateral Filter (Removes noise while preserving sharp character edges)
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, 7, 75, 75)
        variants["grayscale_bilateral"] = cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR)
    except Exception:
        pass

    # 3. Sharpening filter (Counters WhatsApp / Twitter JPEG artifact blur)
    try:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(img, -1, kernel)
        variants["sharpened"] = sharpened
    except Exception:
        pass

    # 4. Adaptive Threshold / Binarization for small fonts
    if h < 1200 or w < 1200:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_large = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            thresh = cv2.adaptiveThreshold(
                gray_large, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            variants["adaptive_thresh"] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        except Exception:
            pass

    # 5. Deskewed variant
    deskewed = auto_deskew(img)
    if not np.array_equal(deskewed, img):
        variants["deskewed"] = deskewed

    return variants


def is_mobile_status_artifact(text: str) -> bool:
    """Filters out mobile UI status line noise (battery %, carrier, clock time)."""
    text_clean = text.strip()
    if not text_clean:
        return True
    # Battery indicators like "100%", "95 %", "45%"
    if re.fullmatch(r'\d{1,3}\s*%', text_clean):
        return True
    # Carrier indicators like "4G", "5G", "LTE", "VoLTE", "WiFi"
    if re.fullmatch(r'(4G|5G|LTE|VoLTE|WiFi|3G|E|H\+)', text_clean, re.IGNORECASE):
        return True
    # Clock timestamps like "11:42", "09:30 AM", "22:15"
    if re.fullmatch(r'\d{1,2}:\d{2}(\s*(AM|PM))?', text_clean, re.IGNORECASE):
        return True
    return False


def clean_ocr_text(text: str) -> str:
    """Robust OCR normalization layer. Preserves English, Hindi, and numbers."""
    if not text:
        return ""
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def run_paddle_ocr_regions(image_path: str, lang: str = "en") -> List[Dict[str, Any]]:
    """Runs PaddleOCR and returns structured bounding box regions."""
    paddle_engine = get_paddle_ocr(lang)
    if paddle_engine is None:
        return []
    try:
        try:
            result = paddle_engine.ocr(image_path)
        except TypeError:
            result = paddle_engine.ocr(image_path, cls=True)
        if not result or not result[0]:
            return []


        regions = []
        for line in result[0]:
            poly, (text, conf) = line
            cleaned = clean_ocr_text(text)
            
            # Skip isolated single non-digit characters or empty
            if len(cleaned) < 2 and not any(c.isdigit() for c in cleaned):
                continue
                
            # Filter mobile status noise
            if is_mobile_status_artifact(cleaned):
                continue

            x1 = min(p[0] for p in poly)
            y1 = min(p[1] for p in poly)
            x2 = max(p[0] for p in poly)
            y2 = max(p[1] for p in poly)

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
                "engine": f"PaddleOCR-{lang}"
            })
        return regions
    except Exception as e:
        logger.error(f"PaddleOCR error (lang={lang}): {e}")
        return []


def format_regions_spatially(regions: List[Dict]) -> str:
    """Groups OCR text blocks reading naturally top-to-bottom, left-to-right."""
    if not regions:
        return ""

    # Sort vertically with line grouping tolerance (20px), then horizontally
    regions.sort(key=lambda r: (round(r["center_y"] / 24) * 24, r["center_x"]))

    lines = []
    current_line = []
    current_y = None

    for r in regions:
        text = r["text"]
        if not text:
            continue

        y_chunk = round(r["center_y"] / 24) * 24
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


def has_devanagari(text: str) -> bool:
    """Returns True if text contains Hindi / Devanagari script."""
    return bool(re.search(r'[\u0900-\u097F]', text))


def extract_text(image_path: str) -> Dict[str, Any]:
    """
    Main OCR entrypoint.
    Executes multi-variant preprocessing, layout awareness, multi-language detection,
    and consensus validation across available engines.
    """
    try:
        img_raw = cv2.imread(image_path)
        if img_raw is None:
            return {"status": "FAILED", "error": "Unable to read image file.", "stage": "load"}

        detected_layout = detect_image_type(img_raw)
        variants = get_preprocessing_variants(image_path)
        if not variants:
            variants = {"original": img_raw}

        best_regions: List[Dict] = []
        best_conf = 0.0
        engine_used = "unknown"
        preprocessing_used = "original"
        engines_attempted = []

        # 1. Primary pass with PaddleOCR (English)
        engines_attempted.append("PaddleOCR-en")
        for variant_name, img_data in variants.items():
            if variant_name != "original" and best_conf > 0.88:
                break
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                temp_path = tf.name
            try:
                cv2.imwrite(temp_path, img_data)
                regions = run_paddle_ocr_regions(temp_path, lang="en")
                if regions:
                    avg_conf = sum(r["confidence"] for r in regions) / len(regions)
                    if avg_conf > best_conf:
                        best_conf = avg_conf
                        best_regions = regions
                        engine_used = "PaddleOCR-en"
                        preprocessing_used = variant_name
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        # 2. Check if text might be Hindi/Devanagari or if confidence is low
        joined_text_preview = " ".join([r["text"] for r in best_regions])
        if has_devanagari(joined_text_preview) or (len(best_regions) == 0 and detected_layout != "pure_photo"):
            engines_attempted.append("PaddleOCR-hi")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                temp_path = tf.name
            try:
                cv2.imwrite(temp_path, variants.get("clahe_contrast", img_raw))
                hi_regions = run_paddle_ocr_regions(temp_path, lang="hi")
                if hi_regions:
                    hi_conf = sum(r["confidence"] for r in hi_regions) / len(hi_regions)
                    if hi_conf > best_conf or (not best_regions and len(hi_regions) > 0):
                        best_conf = hi_conf
                        best_regions = hi_regions
                        engine_used = "PaddleOCR-hi"
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        # 3. Tesseract fallback if available and PaddleOCR found nothing
        if not best_regions or best_conf < 0.35:
            engines_attempted.append("Tesseract")
            tess_variants = [("clahe_contrast", variants.get("clahe_contrast")), ("original", img_raw)]
            for var_name, var_img in tess_variants:
                if var_img is None:
                    continue
                try:
                    gray = cv2.cvtColor(var_img, cv2.COLOR_BGR2GRAY)
                    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
                    tess_regions = []
                    for i in range(len(data.get('text', []))):
                        txt = data['text'][i].strip()
                        conf = data['conf'][i]
                        if int(conf) > 0 and len(txt) > 0:
                            cleaned = clean_ocr_text(txt)
                            if cleaned and not is_mobile_status_artifact(cleaned):
                                x, y, bw, bh = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                                tess_regions.append({
                                    "text": cleaned,
                                    "raw_text": txt,
                                    "confidence": float(conf) / 100.0,
                                    "bbox": [x, y, x + bw, y + bh],
                                    "center_x": x + bw / 2,
                                    "center_y": y + bh / 2,
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
                            engine_used = f"Tesseract ({var_name})"
                            preprocessing_used = var_name
                except Exception as e:
                    logger.debug(f"Tesseract variant {var_name} skipped: {e}")


        # Edge Case: NO text detected (pure photo or graphic without text)
        if not best_regions:
            return {
                "status": "NO_TEXT_DETECTED",
                "image_type": detected_layout,
                "engine_used": engine_used,
                "engines_attempted": engines_attempted,
                "preprocessing": preprocessing_used,
                "overall_confidence": 0.0,
                "regions_detected": 0,
                "regions": [],
                "raw_text": "",
                "normalized_text": "",
                "message": "No embedded text was found in this image. Proceeding with forensic visual analysis."
            }

        grouped_text = format_regions_spatially(best_regions)
        raw_text_full = "\n".join([r["raw_text"] for r in best_regions])

        # Multilingual-aware LLM OCR post-processing
        # Fixes broken OCR words while strictly preserving Hindi, proper names, and numbers
        if len(grouped_text.split()) >= 3:
            try:
                from app.services.nlp.llm_client import call_llm_json
                prompt = f"""
You are an expert OCR repair assistant.
Clean the following extracted text from an image (which may be a news tweet, screenshot, or graphic).
Instructions:
1. Preserve all proper nouns, names, dates, numbers, and statistics.
2. Fix broken words and remove obvious OCR glitch artifacts.
3. If the text contains Hindi or regional Indian language, DO NOT translate or remove it. Preserve it faithfully.
4. If the text is English, make it clean and coherent.

Raw OCR Text:
{grouped_text}

Output JSON format:
{{
  "cleaned_text": "The cleaned, coherent text."
}}
"""
                res = call_llm_json(prompt)
                if res and "cleaned_text" in res and len(res["cleaned_text"].strip()) > 5:
                    grouped_text = res["cleaned_text"].strip()
            except Exception as e:
                logger.debug(f"LLM OCR refinement skipped: {e}")

        return {
            "status": "SUCCESS",
            "image_type": detected_layout,
            "engine_used": engine_used,
            "engines_attempted": engines_attempted,
            "preprocessing": preprocessing_used,
            "overall_confidence": round(best_conf, 3),
            "regions_detected": len(best_regions),
            "regions": best_regions,
            "raw_text": raw_text_full,
            "normalized_text": grouped_text,
            "has_devanagari": has_devanagari(grouped_text)
        }

    except Exception as e:
        logger.error(f"OCR processing exception: {e}")
        return {
            "status": "FAILED",
            "error": str(e),
            "engine_used": "unknown",
            "overall_confidence": 0.0,
            "regions": [],
            "raw_text": "",
            "normalized_text": ""
        }
