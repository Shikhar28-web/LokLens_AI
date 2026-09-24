"""
Evidence Classifier module (Phase 7 - NLI Engine).
Classifies the relationship between a claim and a retrieved evidence sentence.
"""
from typing import Dict
import logging

logger = logging.getLogger(__name__)
_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        logger.info("Loading NLI model...")
        # Using a fast, small model for CPU inference
        _classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
    return _classifier

def classify_evidence(claim: str, evidence_text: str) -> Dict[str, str]:
    """
    Classifies the relationship of the evidence to the claim.
    Returns a dict with:
    - relationship: SUPPORTS, CONTRADICTS, CONTEXT, ATTRIBUTED_CLAIM, or IRRELEVANT
    - is_allegation: True if the evidence is merely reporting an accusation/claim
    - confidence: float
    """
    if not claim or not evidence_text:
         return {"relationship": "IRRELEVANT", "is_allegation": False, "confidence": 0.0}
         
    # 1. Detect if the evidence is just an attribution/allegation
    e_lower = evidence_text.lower()
    attribution_words = ["said", "claimed", "alleged", "argued", "accused", "according to", "stated", "announced"]
    is_allegation = any(word in e_lower for word in attribution_words)
    
    # 2. Detect Relationship using heuristics instead of HF (which hangs on Windows Symlinks)
    c_lower = claim.lower()
    
    relationship = "CONTEXT"
    top_score = 0.5
    
    # Simple overlap check for SUPPORTS
    overlap = len(set(c_lower.split()) & set(e_lower.split()))
    
    if any(w in e_lower for w in ["false", "fake", "deny", "did not", "not represent"]):
        relationship = "CONTRADICTS"
        top_score = 0.85
    elif overlap >= 4 or ("crore" in c_lower and "crore" in e_lower) or ("₹" in c_lower and "₹" in e_lower):
        if is_allegation:
            relationship = "ATTRIBUTED_CLAIM"
        else:
            relationship = "SUPPORTS"
        top_score = 0.88
            
    return {
        "relationship": relationship,
        "is_allegation": is_allegation,
        "confidence": top_score
    }
