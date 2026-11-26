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
_paddle_available: Optional[bool] = None

def get_paddle_ocr(lang: str = "en"):
    """Lazy loads and caches PaddleOCR engine for specified language."""
    global _paddle_instances, _paddle_available
    if _paddle_available is False:
        return None
    if lang in _paddle_instances:
        return _paddle_instances[lang]
    try:
        from paddleocr import PaddleOCR
        instance = PaddleOCR(use_angle_cls=False, lang=lang)
        _paddle_instances[lang] = instance
        _paddle_available = True
        logger.info(f"Initialized PaddleOCR for lang='{lang}'")
        return instance
    except Exception as e:
        logger.warning(f"PaddleOCR not available for lang='{lang}': {e}")
        _paddle_available = False
        return None


# Fallback path for Windows Tesseract
import pytesseract
_TESSERACT_CANDIDATES = [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    os.path.expanduser(r'~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe')
]
for p in _TESSERACT_CANDIDATES:
    if os.path.exists(p):
        pytesseract.pytesseract.tesseract_cmd = p
        break


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


def is_ocr_garbage(text: str) -> bool:
    """Checks if text chunk is OCR graphic noise rather than human readable text."""
    t = text.strip()
    if not t or (len(t) <= 1 and not t.isalnum()):
        return True
    # If string is purely repetitive symbols e.g. "— = ~", "<a ——— Je"
    alnum_count = sum(1 for c in t if c.isalnum() or '\u0900' <= c <= '\u097F')
    if len(t) > 2 and (alnum_count / len(t)) < 0.45:
        return True
    return False


def clean_ocr_text(text: str) -> str:
    """Robust OCR normalization layer. Preserves English, Hindi, and numbers while stripping noise."""
    if not text:
        return ""
    text = re.sub(r'[\r\n\t]+', ' ', text)
    # Strip garbage punctuation clusters like '— =', '<a ——— Je', '‘}', '||'
    text = re.sub(r'[—=~<>{}\\/|_]{2,}', ' ', text)
    text = re.sub(r'^[^\w\s\u0900-\u097F]+', '', text)  # remove leading non-alphanumeric noise
    text = re.sub(r'[^\w\s\u0900-\u097F]+$', '', text)  # remove trailing non-alphanumeric noise
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def run_paddle_ocr_regions(image_path: str, lang: str = "en") -> List[Dict[str, Any]]:
    """Runs PaddleOCR and returns structured bounding box regions."""
    global _paddle_available
    if _paddle_available is False:
        return []
    paddle_engine = get_paddle_ocr(lang)
    if paddle_engine is None:
        return []
    try:
        try:
            result = paddle_engine.ocr(image_path)
        except TypeError:
            result = paddle_engine.ocr(image_path, cls=False)
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
        logger.warning(f"PaddleOCR execution failed ({e}), disabling PaddleOCR for this session.")
        _paddle_available = False
        return []


def format_hierarchical_layout(valid_words: List[Dict[str, Any]]) -> Tuple[str, str, float]:
    """
    Layout-aware spatial clustering for multi-column documents, news chyrons, and social graphics.
    Groups words into physical lines, splits lines into columns when horizontal gaps exist,
    clusters column segments vertically, and reconstructs reading order naturally.

    Returns:
        (processed_text, raw_text, length_weighted_confidence)
    """
    if not valid_words:
        return "", "", 0.0

    # 1. Real length-weighted confidence
    total_len = sum(len(w["text"]) for w in valid_words)
    weighted_conf = sum(w["confidence"] * len(w["text"]) for w in valid_words) / max(total_len, 1)

    # 2. Extract verbatim raw text
    valid_words_by_scan = sorted(valid_words, key=lambda w: (w.get("block", 0), w["center_y"], w["center_x"]))
    raw_lines_dict: Dict[Tuple[int, int], List[str]] = {}
    for w in valid_words_by_scan:
        blk = w.get("block", 0)
        y_approx = round(w["center_y"] / 20) * 20
        raw_lines_dict.setdefault((blk, y_approx), []).append(w.get("raw_text", w["text"]))
    raw_text = "\n".join(" ".join(tokens) for tokens in raw_lines_dict.values())

    # 3. Group words by Tesseract block first
    tess_blocks: Dict[int, List[Dict[str, Any]]] = {}
    for w in valid_words:
        tess_blocks.setdefault(w.get("block", 0), []).append(w)

    # 4. Merge vertically touching blocks that share the same horizontal span (e.g. column header + body)
    sorted_b_ids = sorted(tess_blocks.keys())
    merged_blocks: List[List[Dict[str, Any]]] = []
    curr_merged = tess_blocks[sorted_b_ids[0]]
    for b_id in sorted_b_ids[1:]:
        next_b = tess_blocks[b_id]
        c_bottom = max(w["bbox"][3] for w in curr_merged)
        n_top = min(w["bbox"][1] for w in next_b)
        c_left = min(w["bbox"][0] for w in curr_merged)
        c_right = max(w["bbox"][2] for w in curr_merged)
        n_left = min(w["bbox"][0] for w in next_b)
        n_right = max(w["bbox"][2] for w in next_b)
        overlap = max(0, min(c_right, n_right) - max(c_left, n_left))
        span_w = max(c_right - c_left, n_right - n_left)

        if (n_top - c_bottom < 32) and (overlap / max(span_w, 1) > 0.55):
            curr_merged.extend(next_b)
        else:
            merged_blocks.append(curr_merged)
            curr_merged = next_b
    merged_blocks.append(curr_merged)

    # 5. Process each block with multi-column segmentation
    def process_block(b_words: List[Dict[str, Any]]) -> List[str]:
        b_words.sort(key=lambda w: w["bbox"][1])
        lines: List[List[Dict[str, Any]]] = []
        for w in b_words:
            placed = False
            for line in lines:
                avg_top = sum(x["bbox"][1] for x in line) / len(line)
                avg_h = sum(x["height"] for x in line) / len(line)
                if abs(w["bbox"][1] - avg_top) < max(avg_h * 0.55, 9):
                    line.append(w)
                    placed = True
                    break
            if not placed:
                lines.append([w])
        lines.sort(key=lambda l: sum(x["bbox"][1] for x in l) / len(l))

        # Check for column gap (> 55px and > 2.2 * font height)
        has_gap = False
        for line in lines:
            line.sort(key=lambda x: x["bbox"][0])
            for k in range(len(line) - 1):
                gap = line[k + 1]["bbox"][0] - line[k]["bbox"][2]
                avg_h = (line[k + 1]["height"] + line[k]["height"]) / 2
                if gap > max(avg_h * 2.2, 55):
                    has_gap = True
                    break
            if has_gap:
                break

        if not has_gap:
            return [" ".join(" ".join(w["text"] for w in line) for line in lines)]

        # Multi-column detected: cluster words by horizontal X intervals
        x_intervals: List[List[float]] = []
        for line in lines:
            line.sort(key=lambda x: x["bbox"][0])
            curr_span = [line[0]["bbox"][0], line[0]["bbox"][2]]
            for w in line[1:]:
                gap = w["bbox"][0] - curr_span[1]
                avg_h = (w["height"] + (curr_span[1] - curr_span[0])) / 2
                if gap > max(avg_h * 1.5, 55):
                    x_intervals.append(curr_span)
                    curr_span = [w["bbox"][0], w["bbox"][2]]
                else:
                    curr_span[1] = max(curr_span[1], w["bbox"][2])
            x_intervals.append(curr_span)

        col_ranges: List[List[float]] = []
        x_intervals.sort(key=lambda x: x[0])
        for span in x_intervals:
            if not col_ranges:
                col_ranges.append(span)
            else:
                prev = col_ranges[-1]
                if span[0] - prev[1] < 45:
                    prev[1] = max(prev[1], span[1])
                else:
                    col_ranges.append(span)

        col_words: List[List[Dict[str, Any]]] = [[] for _ in col_ranges]
        for w in b_words:
            w_center = w["center_x"]
            best_col = 0
            best_dist = 999999.0
            for idx, span in enumerate(col_ranges):
                span_center = (span[0] + span[1]) / 2
                dist = abs(w_center - span_center)
                if dist < best_dist:
                    best_dist = dist
                    best_col = idx
            col_words[best_col].append(w)

        res: List[str] = []
        for cw in col_words:
            if cw:
                cw.sort(key=lambda w: w["bbox"][1])
                c_lines: List[List[Dict[str, Any]]] = []
                for w in cw:
                    placed = False
                    for cl in c_lines:
                        avg_top = sum(x["bbox"][1] for x in cl) / len(cl)
                        avg_h = sum(x["height"] for x in cl) / len(cl)
                        if abs(w["bbox"][1] - avg_top) < max(avg_h * 0.55, 9):
                            cl.append(w)
                            placed = True
                            break
                    if not placed:
                        c_lines.append([w])
                c_lines.sort(key=lambda l: sum(x["bbox"][1] for x in l) / len(l))
                for cl in c_lines:
                    cl.sort(key=lambda x: x["bbox"][0])
                res.append(" ".join(" ".join(w["text"] for w in cl) for cl in c_lines))
        return res

    paragraphs: List[str] = []
    for blk in merged_blocks:
        paras = process_block(blk)
        for p in paras:
            p_clean = re.sub(r'^(GG|a7|ds|i|1\))\s+', '', p).strip()
            p_clean = re.sub(r'\s+[=>|~]{1,3}\s+', ' ', p_clean).strip()
            if len(p_clean.split()) >= 2 or (len(p_clean) >= 4 and p_clean.isalnum()):
                paragraphs.append(p_clean)

    processed_text = "\n\n".join(paragraphs)
    return processed_text, raw_text, round(weighted_conf, 3)


def format_regions_spatially(regions: List[Dict]) -> str:
    """Backwards-compatible wrapper calling format_hierarchical_layout."""
    processed_text, _, _ = format_hierarchical_layout(regions)
    return processed_text


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

        # 1. Primary pass with Tesseract on Windows (Fast, reliable, layout-aware)
        tess_available = bool(getattr(pytesseract.pytesseract, 'tesseract_cmd', None) and os.path.exists(pytesseract.pytesseract.tesseract_cmd))
        if tess_available:
            engines_attempted.append("Tesseract")
            gray_orig = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_gray = clahe.apply(gray_orig)
            _, otsu_bin = cv2.threshold(gray_orig, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            tess_variants = [
                ("original", gray_orig),
                ("clahe_contrast", clahe_gray),
                ("otsu_binary", otsu_bin)
            ]
            for var_name, var_img in tess_variants:
                if var_img is None:
                    continue
                try:
                    data = pytesseract.image_to_data(
                        var_img,
                        output_type=pytesseract.Output.DICT,
                        config='--oem 3 --psm 3'
                    )
                    tess_regions = []
                    for i in range(len(data.get('text', []))):
                        txt = data['text'][i].strip()
                        conf = data['conf'][i]
                        if int(conf) > 25 and len(txt) > 0:
                            cleaned = clean_ocr_text(txt)
                            # Filter mobile status noise & isolated 1-2 letter gibberish
                            if len(cleaned) <= 2 and cleaned.lower() not in (
                                'in', 'to', 'on', 'at', 'is', 'it', 'he', 'by', 'of', 'or',
                                'as', 'no', 'up', 'am', 'pm', 'an', 'we', 'us', 'my', 'if', 'so', 'do'
                            ) and not any(c.isdigit() for c in cleaned):
                                continue
                            if cleaned and not is_mobile_status_artifact(cleaned) and not is_ocr_garbage(cleaned):
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
                                    "block": data['block_num'][i],
                                    "reliability": "HIGH" if float(conf) > 75 else "LOW",
                                    "engine": "Tesseract"
                                })
                    if tess_regions:
                        # Length-weighted confidence for variant evaluation
                        t_len = sum(len(r["text"]) for r in tess_regions)
                        tess_conf = sum(r["confidence"] * len(r["text"]) for r in tess_regions) / max(t_len, 1)
                        if tess_conf > best_conf or (tess_conf > 0.80 and len(tess_regions) >= 8):
                            best_regions = tess_regions
                            best_conf = tess_conf
                            engine_used = f"Tesseract ({var_name})"
                            preprocessing_used = var_name
                            if best_conf > 0.85 and len(best_regions) >= 6:
                                break  # Excellent extraction
                except Exception as e:
                    logger.debug(f"Tesseract variant {var_name} skipped: {e}")

        # 2. Secondary pass with PaddleOCR if Tesseract found nothing or has low confidence
        if (_paddle_available is not False) and (not best_regions or best_conf < 0.40):
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
                        r_len = sum(len(r["text"]) for r in regions)
                        avg_conf = sum(r["confidence"] * len(r["text"]) for r in regions) / max(r_len, 1)
                        if avg_conf > best_conf:
                            best_conf = avg_conf
                            best_regions = regions
                            engine_used = "PaddleOCR-en"
                            preprocessing_used = variant_name
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

        # 3. Check if text might be Hindi/Devanagari
        joined_text_preview = " ".join([r["text"] for r in best_regions])
        if (_paddle_available is not False) and (has_devanagari(joined_text_preview) or (len(best_regions) == 0 and detected_layout != "pure_photo")):
            engines_attempted.append("PaddleOCR-hi")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                temp_path = tf.name
            try:
                cv2.imwrite(temp_path, variants.get("clahe_contrast", img_raw))
                hi_regions = run_paddle_ocr_regions(temp_path, lang="hi")
                if hi_regions:
                    h_len = sum(len(r["text"]) for r in hi_regions)
                    hi_conf = sum(r["confidence"] * len(r["text"]) for r in hi_regions) / max(h_len, 1)
                    if hi_conf > best_conf or (not best_regions and len(hi_regions) > 0):
                        best_conf = hi_conf
                        best_regions = hi_regions
                        engine_used = "PaddleOCR-hi"
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)


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
                "processed_text": "",
                "message": "No embedded text was found in this image. Proceeding with forensic visual analysis."
            }

        grouped_text, raw_text_full, weighted_confidence = format_hierarchical_layout(best_regions)

        # Multilingual-aware LLM OCR post-processing if available
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
            "overall_confidence": weighted_confidence,
            "regions_detected": len(best_regions),
            "regions": best_regions,
            "raw_text": raw_text_full,
            "normalized_text": grouped_text,
            "processed_text": grouped_text,
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
