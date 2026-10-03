import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def estimate_ai_likelihood(scores: Dict[str, float]) -> Dict[str, Any]:
    """
    Estimates the likelihood that an image is AI generated or manipulated,
    based on a combination of forensic scores.
    """
    ela_score = scores.get("ela", 0.0)
    noise_score = scores.get("noise", 0.0)
    frequency_score = scores.get("frequency", 0.0)
    copy_move_score = scores.get("copy_move", 0.0)
    
    # AI generators typically leave strong frequency anomalies, but minimal ELA anomalies.
    # Manual photoshop leaves strong ELA and noise anomalies, but lower frequency anomalies.
    
    # Check if this is likely a screenshot/infographic (where frequency score breaks due to text)
    # If the frequency score is extremely high but ELA and noise are negligible, it's almost certainly text/sharp edges.
    is_text_heavy = frequency_score > 0.8 and ela_score < 0.15 and noise_score < 0.15
    
    if is_text_heavy:
        return {
            "ai_likelihood_score": None,
            "ai_likelihood_label": "UNKNOWN",
            "model": "rule_based_calibration"
        }
    
    # Weight the scores.
    # High frequency score strongly correlates with GAN/Diffusion models.
    ai_score = (frequency_score * 0.5) + (noise_score * 0.25) + (ela_score * 0.25)
    ai_score = min(ai_score, 1.0)
    
    # Determine label
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
        "model": "heuristic_ensemble_v1"
    }
