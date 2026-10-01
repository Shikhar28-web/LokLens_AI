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
    
    # Weight the scores.
    # High frequency score strongly correlates with GAN/Diffusion models.
    ai_score = (frequency_score * 0.6) + (noise_score * 0.2) + (ela_score * 0.2)
    ai_score = min(ai_score, 1.0)
    
    # Determine label (from Verdict Engine Logic in Phase 12 plan)
    if ai_score >= 0.75:
        label = "likely_ai_generated"
    elif ai_score >= 0.55 or (ai_score >= 0.4 and frequency_score > 0.6):
        label = "possibly_ai_generated"
    elif ela_score >= 0.75 or copy_move_score >= 0.6:
        label = "likely_manipulated"
    elif ela_score >= 0.6 or noise_score >= 0.6 or copy_move_score >= 0.4:
        label = "possibly_manipulated"
    elif ai_score < 0.2 and ela_score < 0.2 and noise_score < 0.2 and copy_move_score < 0.2:
        label = "likely_authentic"
    elif ai_score < 0.35 and max(ela_score, noise_score, copy_move_score) < 0.35:
        label = "possibly_authentic"
    else:
        label = "inconclusive"
        
    return {
        "ai_likelihood_score": round(ai_score, 4),
        "ai_likelihood_label": label
    }
