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

def check_source_relevance(claim: str, document_text: str) -> dict:
    """
    Strict source relevance/validation layer.
    Compares source against the claim using entity overlap, keywords, and topic matching.
    """
    c_lower = claim.lower()
    d_lower = document_text.lower()
    
    # 1. Very strict check: If claim mentions a specific location/entity, document must mention it
    from app.services.nlp.ner import extract_entities
    claim_ents = extract_entities(claim)
    
    # Check locations
    missing_locs = []
    for loc in claim_ents.get("locations", []):
        if loc.lower() not in d_lower:
            missing_locs.append(loc)
            
    if missing_locs and len(missing_locs) == len(claim_ents.get("locations", [])):
        return {"relevant": False, "score": 0.0, "reason": f"Source is missing key locations: {missing_locs}"}
        
    # Check key ORGs
    missing_orgs = []
    for org in claim_ents.get("orgs", []):
        # Allow partial match for long org names
        org_words = org.lower().split()
        if not any(w in d_lower for w in org_words if len(w) > 3):
            missing_orgs.append(org)
            
    if missing_orgs and len(missing_orgs) == len(claim_ents.get("orgs", [])):
        return {"relevant": False, "score": 0.0, "reason": f"Source is missing key organizations: {missing_orgs}"}
        
    # Semantic keyword overlap
    c_words = set([w for w in c_lower.split() if len(w) > 4])
    d_words = set(d_lower.split())
    
    overlap = len(c_words & d_words)
    if len(c_words) > 0:
        ratio = overlap / len(c_words)
    else:
        ratio = 1.0
        
    if ratio < 0.2 and overlap < 3:
        return {"relevant": False, "score": ratio, "reason": "Source has insufficient keyword overlap with the claim."}
        
    # Reject generic dictionary/wiki definitions for event-based claims
    # e.g., if claim has numbers/specifics, but source is just generic entity intro
    import re
    claim_has_numbers = bool(re.search(r'\d+', claim))
    if claim_has_numbers and not re.search(r'\d+', document_text):
        return {"relevant": False, "score": ratio, "reason": "Claim involves numbers/statistics not found in source."}
        
    return {"relevant": True, "score": ratio, "reason": "Source matches claim entities and topics."}

def classify_evidence(claim: str, evidence_text: str) -> Dict[str, str]:
    """
    Classifies the relationship of the evidence to the claim.
    Extracts the exact supporting/contradicting reason.
    """
    if not claim or not evidence_text:
         return {"relationship": "IRRELEVANT", "is_allegation": False, "confidence": 0.0, "reason": "No text provided."}
         
    e_lower = evidence_text.lower()
    c_lower = claim.lower()
    
    attribution_words = ["said", "claimed", "alleged", "argued", "accused", "according to", "stated", "announced"]
    is_allegation = any(word in e_lower for word in attribution_words)
    
    relationship = "CONTEXT"
    top_score = 0.5
    reason = "The source provides related context but does not explicitly confirm or deny the claim."
    
    # Check for IRRELEVANT first
    # If the evidence text doesn't contain any of the core nouns/numbers from the claim, it's irrelevant.
    import re
    c_words = [w for w in c_lower.split() if len(w) > 3]
    overlap_count = sum(1 for w in c_words if w in e_lower)
    
    if overlap_count < 2 and not any(re.findall(r'\d+', c_lower)) == any(re.findall(r'\d+', e_lower)):
        return {
            "relationship": "IRRELEVANT",
            "is_allegation": False,
            "confidence": 0.9,
            "reason": "The text does not meaningfully address the specifics of the claim."
        }
    
    overlap = len(set(c_lower.split()) & set(e_lower.split()))
    if any(w in e_lower for w in ["false", "fake", "deny", "did not", "not represent", "refuted", "misleading"]):
        relationship = "CONTRADICTS"
        top_score = 0.85
        reason = f"The source explicitly contradicts the claim using terms like 'false' or 'deny'."
    elif overlap >= 4 or ("crore" in c_lower and "crore" in e_lower) or ("₹" in c_lower and "₹" in e_lower) or any(n in e_lower for n in re.findall(r'\d+', c_lower) if n):
        if is_allegation:
            relationship = "ATTRIBUTED_CLAIM"
            reason = "The source reports that someone alleged this claim, but does not independently verify it."
        else:
            relationship = "SUPPORTS"
            reason = "The source explicitly states facts that align with the core claim."
        top_score = 0.88
            
    return {
        "relationship": relationship,
        "is_allegation": is_allegation,
        "confidence": top_score,
        "reason": reason
    }
