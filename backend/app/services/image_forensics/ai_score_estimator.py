import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def estimate_ai_likelihood(scores: Dict[str, float]) -> Dict[str, Any]:
    """
    Estimates the likelihood that an image is AI generated or manipulated,
    based on a combination of forensic scores.

    v2 improvements:
    - Extended text-heavy / graphic-overlay guard (frequency > 0.60 threshold)
    - Adaptive weight scheme: reduces FFT weight from 0.50 → 0.25 for likely
      text/graphic images to prevent false positives on screenshots and memes
    - EXIF authenticity boost: if metadata confirms a real camera, apply -0.10
      to the final composite score
    """
    ela_score = scores.get("ela", 0.0)
    noise_score = scores.get("noise", 0.0)
    frequency_score = scores.get("frequency", 0.0)
    copy_move_score = scores.get("copy_move", 0.0)
    # Optional: EXIF-based camera authenticity boost (-0.10 if real camera detected)
    exif_boost = scores.get("exif_authenticity_boost", 0.0)

    # ── Guard 1: Pure text / infographic (original guard) ────────────────────
    # Very high FFT but nearly zero ELA + noise → sharp computer-generated fonts
    is_text_heavy = frequency_score > 0.80 and ela_score < 0.15 and noise_score < 0.15

    if is_text_heavy:
        return {
            "ai_likelihood_score": None,
            "ai_likelihood_label": "UNKNOWN",
            "model": "rule_based_calibration"
        }

    # ── Guard 2: Mixed content (photo + text overlay, news graphics, memes) ──
    # Moderate FFT + low ELA + low copy-move → reduce FFT weight heavily
    is_text_graphic_mix = (
        frequency_score > 0.60
        and ela_score < 0.10
        and copy_move_score < 0.10
    )

    if is_text_graphic_mix:
        # Adaptive weights: ELA and noise dominate, FFT de-weighted
        ai_score = (frequency_score * 0.25) + (noise_score * 0.35) + (ela_score * 0.40)
    else:
        # Standard weights: high FFT strongly correlates with GAN/Diffusion models
        ai_score = (frequency_score * 0.50) + (noise_score * 0.25) + (ela_score * 0.25)

    # Apply EXIF camera authenticity boost (negative → lowers AI score)
    ai_score = max(0.0, min(ai_score + exif_boost, 1.0))

    # ── Determine verdict label ───────────────────────────────────────────────
    if ai_score >= 0.75:
        label = "LIKELY_AI_GENERATED"
    elif ai_score >= 0.55 or (ai_score >= 0.4 and frequency_score > 0.6):
        label = "POSSIBLY_AI_GENERATED"
    elif ela_score >= 0.75 or copy_move_score >= 0.6:
        label = "LIKELY_MANIPULATED"
    elif ela_score >= 0.6 or noise_score >= 0.6 or copy_move_score >= 0.4:
        label = "POSSIBLY_MANIPULATED"
    elif ai_score < 0.2 and ela_score < 0.2 and noise_score < 0.2 and copy_move_score < 0.2:
        label = "LIKELY_AUTHENTIC"
    elif ai_score < 0.35 and max(ela_score, noise_score, copy_move_score) < 0.35:
        label = "POSSIBLY_AUTHENTIC"
    else:
        label = "UNKNOWN"

    if label == "UNKNOWN":
        ai_score = None

    return {
        "ai_likelihood_score": round(ai_score, 4) if ai_score is not None else None,
        "ai_likelihood_label": label,
        "model": "heuristic_ensemble_v2"
    }
