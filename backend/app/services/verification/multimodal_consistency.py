"""
Phase 15: Multimodal Consistency Analysis

Compares image signals against text signals to detect inconsistencies:
  - OCR text from image  <->  claim text entity overlap
  - Image EXIF date      <->  dates mentioned in claims
  - pHash match verdict  <->  source credibility
  - Viral recirculation  <->  claim recency
"""

import re
import logging
from typing import Optional, List, Dict, Any
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def _extract_years(text: str) -> List[int]:
    """Pull 4-digit years from a string."""
    return [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', text)]


def _text_overlap_score(text_a: str, text_b: str) -> float:
    """Return a 0-1 overlap score using SequenceMatcher on lowercased text."""
    if not text_a or not text_b:
        return 0.0
    a = text_a.lower()
    b = text_b.lower()
    return SequenceMatcher(None, a, b).ratio()


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    if not text_a or not text_b:
        return 0.0
    stop = {"the", "a", "an", "is", "in", "on", "at", "to", "of", "and", "or", "for",
            "with", "that", "this", "it", "as", "was", "be", "by", "from"}
    tok_a = {w for w in re.findall(r'\w+', text_a.lower()) if len(w) > 2 and w not in stop}
    tok_b = {w for w in re.findall(r'\w+', text_b.lower()) if len(w) > 2 and w not in stop}
    if not tok_a or not tok_b:
        return 0.0
    intersection = tok_a & tok_b
    union = tok_a | tok_b
    return len(intersection) / len(union)


def analyze_multimodal_consistency(
    ocr_text: Optional[str],
    claim_texts: List[str],
    raw_input_text: Optional[str],
    phash_match_verdict: Optional[str],
    exif_json: Optional[Dict[str, Any]],
    ai_likelihood_label: Optional[str],
    noise_anomaly: Optional[float],
    ela_score: Optional[float],
) -> Dict[str, Any]:
    """
    Produces a consistency score (0–1) and a set of flags.

    High score  → image and text are consistent with each other.
    Low score   → significant mismatch detected (e.g. old image, wrong location).
    """
    signals: Dict[str, Any] = {}
    score_components: List[float] = []
    flags: List[str] = []

    # ── 1. OCR ↔ Claim Text Overlap ─────────────────────────────────────────
    if ocr_text and claim_texts:
        combined_claims = " ".join(claim_texts)
        kw_overlap = _keyword_overlap(ocr_text, combined_claims)
        signals["ocr_claim_keyword_overlap"] = round(kw_overlap, 4)
        score_components.append(kw_overlap)

        if kw_overlap < 0.05:
            flags.append("OCR_TEXT_MISMATCH: Image text shares very few keywords with submitted claim.")
        elif kw_overlap > 0.4:
            flags.append("OCR_TEXT_CONSISTENT: Image text strongly matches the submitted claim.")
    elif ocr_text and raw_input_text:
        kw_overlap = _keyword_overlap(ocr_text, raw_input_text)
        signals["ocr_input_keyword_overlap"] = round(kw_overlap, 4)
        score_components.append(kw_overlap)
    else:
        # No OCR text — neutral contribution
        score_components.append(0.5)
        signals["ocr_claim_keyword_overlap"] = None

    # ── 2. Date Consistency (EXIF date vs. claim dates) ─────────────────────
    exif_year = None
    if exif_json:
        for field in ["DateTimeOriginal", "DateTime", "DateTimeDigitized"]:
            val = exif_json.get(field, "")
            years = _extract_years(str(val))
            if years:
                exif_year = years[0]
                break

    claim_years: List[int] = []
    for ct in claim_texts:
        claim_years.extend(_extract_years(ct))
    if raw_input_text:
        claim_years.extend(_extract_years(raw_input_text))

    if exif_year and claim_years:
        year_diff = min(abs(exif_year - cy) for cy in claim_years)
        if year_diff == 0:
            date_score = 1.0
        elif year_diff <= 1:
            date_score = 0.7
        elif year_diff <= 3:
            date_score = 0.4
            flags.append(f"DATE_MISMATCH: EXIF year {exif_year} differs from claim year(s) {claim_years} by {year_diff} years.")
        else:
            date_score = 0.0
            flags.append(f"DATE_MISMATCH_CRITICAL: EXIF year {exif_year} differs from claim year(s) {claim_years} by {year_diff} years — old image reuse suspected.")
        signals["exif_claim_year_diff"] = year_diff
        score_components.append(date_score)
    else:
        score_components.append(0.5)  # neutral — can't compare
        signals["exif_claim_year_diff"] = None

    # ── 3. Viral Recirculation Penalty ──────────────────────────────────────
    if phash_match_verdict == "VIRAL_RECIRCULATION":
        score_components.append(0.1)
        flags.append("VIRAL_RECIRCULATION: This image has been submitted multiple times — may be misused in different contexts.")
    elif phash_match_verdict == "POSSIBLE_DUPLICATE":
        score_components.append(0.4)
        flags.append("POSSIBLE_DUPLICATE: Image is visually similar to a previously analysed image.")
    else:
        score_components.append(0.8)  # no known duplicate

    # ── 4. AI-Generation Penalty ────────────────────────────────────────────
    ai_penalty_map = {
        "likely_ai_generated": 0.0,
        "possibly_ai_generated": 0.3,
        "possibly_manipulated": 0.3,
        "likely_manipulated": 0.1,
        "possibly_authentic": 0.85,
        "likely_authentic": 1.0,
        "inconclusive": 0.5,
    }
    if ai_likelihood_label:
        ai_score = ai_penalty_map.get(ai_likelihood_label.lower(), 0.5)
        score_components.append(ai_score)
        signals["ai_penalty_applied"] = ai_likelihood_label
        if ai_score < 0.4:
            flags.append(f"AI_GENERATED: Image scored as '{ai_likelihood_label}' — reduces overall consistency.")
    else:
        score_components.append(0.5)

    # ── 5. Compute weighted average ──────────────────────────────────────────
    # Weights: [OCR overlap, date, viral, ai]
    weights = [0.35, 0.20, 0.25, 0.20]
    if len(score_components) < len(weights):
        weights = weights[:len(score_components)]
    total_w = sum(weights)
    consistency_score = sum(s * w for s, w in zip(score_components, weights)) / total_w
    consistency_score = round(min(max(consistency_score, 0.0), 1.0), 4)

    return {
        "multimodal_consistency_score": consistency_score,
        "flags": flags,
        "signals": signals,
    }
