"""
Evidence Classifier & Relevance Validation Module.
Evaluates relationship between claims and retrieved web & social media evidence
(SUPPORTS, CONTRADICTS, ATTRIBUTED_CLAIM, SOCIAL_VIRAL_POST, CONTEXT, IRRELEVANT).
"""

from typing import Dict, Any
import logging
import re

logger = logging.getLogger(__name__)

CONTRADICTION_KEYWORDS = [
    "false", "fake", "deny", "denies", "denied", "debunk", "debunked",
    "hoax", "fabricated", "doctored", "morphed", "misleading", "fact check",
    "no such", "untrue", "busted", "misinformation", "disinformation",
    "refuted", "baseless", "incorrect", "distortion", "claims falsely",
    "not true", "never said", "clarified that no", "distorted"
]

SUPPORT_KEYWORDS = [
    "confirmed", "announced", "approved", "passed", "issued", "held",
    "inaugurated", "declared", "verified", "signed", "stated", "enacted",
    "agreed", "launched", "according to official records", "data shows"
]

ATTRIBUTION_KEYWORDS = [
    "claimed", "alleged", "accused", "argued", "said", "posted", "tweeted",
    "shared", "captioned", "circulating", "viral video shows", "purportedly"
]


def check_source_relevance(claim: str, document_text: str, is_social: bool = False) -> dict:
    """
    Source relevance and validation layer.
    Compares source against the claim using entity overlap, keywords, and semantic matching.
    Social media posts have flexible length and matching criteria.
    """
    if not claim or not document_text:
        return {"relevant": False, "score": 0.0, "reason": "Empty claim or document."}

    c_lower = claim.lower()
    d_lower = document_text.lower()

    # If it's a social media post, ensure minimum meaningful length
    if is_social:
        if len(d_lower.strip()) < 20:
            return {"relevant": False, "score": 0.1, "reason": "Social media post is too short to evaluate."}
        # Check keyword presence
        c_keywords = [w for w in re.findall(r'[\w\u0900-\u097F]+', c_lower) if len(w) > 3]
        overlap_social = sum(1 for w in c_keywords if w in d_lower)
        if overlap_social >= 1 or len(c_keywords) <= 2:
            return {"relevant": True, "score": 0.75, "reason": "Social media post matches core claim subject."}

    # Strict check for standard web articles
    from app.services.nlp.ner import extract_entities
    claim_ents = extract_entities(claim)

    # Check key ORGs
    missing_orgs = []
    for org in claim_ents.get("orgs", []):
        org_words = [w for w in org.lower().split() if len(w) > 3]
        if org_words and not any(w in d_lower for w in org_words):
            missing_orgs.append(org)

    if missing_orgs and len(missing_orgs) == len(claim_ents.get("orgs", [])) and len(claim_ents.get("orgs", [])) > 1:
        return {"relevant": False, "score": 0.0, "reason": f"Source is missing key organizations: {missing_orgs}"}

    # Semantic keyword overlap
    c_words = set(w for w in re.findall(r'[\w\u0900-\u097F]+', c_lower) if len(w) > 3)
    d_words = set(re.findall(r'[\w\u0900-\u097F]+', d_lower))

    overlap = len(c_words & d_words)
    ratio = overlap / len(c_words) if c_words else 1.0

    if ratio < 0.18 and overlap < 2:
        return {"relevant": False, "score": ratio, "reason": "Source has insufficient keyword overlap with the claim."}

    return {"relevant": True, "score": min(ratio + 0.3, 1.0), "reason": "Source matches claim entities and topics."}


def classify_evidence(claim: str, evidence_text: str, is_social: bool = False) -> Dict[str, Any]:
    """
    Classifies relationship between claim and evidence:
    - SUPPORTS
    - CONTRADICTS
    - ATTRIBUTED_CLAIM
    - SOCIAL_VIRAL_POST
    - CONTEXT
    - IRRELEVANT
    """
    if not claim or not evidence_text:
        return {
            "relationship": "IRRELEVANT",
            "is_allegation": False,
            "confidence": 0.0,
            "reason": "No text provided."
        }

    e_lower = evidence_text.lower()
    c_lower = claim.lower()

    is_allegation = any(word in e_lower for word in ATTRIBUTION_KEYWORDS)
    has_contradiction = any(word in e_lower for word in CONTRADICTION_KEYWORDS)
    has_support = any(word in e_lower for word in SUPPORT_KEYWORDS)

    # Keyword overlap count
    c_words = [w for w in re.findall(r'[\w\u0900-\u097F]+', c_lower) if len(w) > 3]
    d_words = set(re.findall(r'[\w\u0900-\u097F]+', e_lower))
    overlap_count = sum(1 for w in c_words if w in d_words)

    # 1. Check for Irrelevant
    if overlap_count < 1 and not is_social:
        return {
            "relationship": "IRRELEVANT",
            "is_allegation": False,
            "confidence": 0.85,
            "reason": "The text does not directly address the specifics of the claim."
        }

    # 2. Strong Contradiction
    if has_contradiction:
        found_kw = [k for k in CONTRADICTION_KEYWORDS if k in e_lower][:2]
        return {
            "relationship": "CONTRADICTS",
            "is_allegation": is_allegation,
            "confidence": 0.90,
            "reason": f"The source explicitly refutes the claim using terms such as '{', '.join(found_kw)}'."
        }

    # 3. Social Media Viral Post
    if is_social:
        if is_allegation or "viral" in e_lower or "post" in e_lower or "video" in e_lower:
            return {
                "relationship": "SOCIAL_VIRAL_POST",
                "is_allegation": True,
                "confidence": 0.80,
                "reason": "This social media post actively circulates or comments on the claim without authoritative verification."
            }
        return {
            "relationship": "CONTEXT",
            "is_allegation": False,
            "confidence": 0.65,
            "reason": "Social media commentary related to this topic."
        }

    # 4. Attributed Claim (someone said it, not verified fact)
    if is_allegation and not has_support:
        return {
            "relationship": "ATTRIBUTED_CLAIM",
            "is_allegation": True,
            "confidence": 0.82,
            "reason": "The source reports that an entity made this allegation, but does not independently establish it as fact."
        }

    # 5. Strong Support
    c_numbers = re.findall(r'\d+', c_lower)
    e_numbers = re.findall(r'\d+', e_lower)
    numbers_match = bool(set(c_numbers) & set(e_numbers)) if c_numbers else False

    if has_support or overlap_count >= 3 or numbers_match or ("₹" in c_lower and "₹" in e_lower):
        return {
            "relationship": "SUPPORTS",
            "is_allegation": False,
            "confidence": 0.88,
            "reason": "The source explicitly reports verified facts and events that align directly with the claim."
        }

    # 6. General Context
    return {
        "relationship": "CONTEXT",
        "is_allegation": is_allegation,
        "confidence": 0.60,
        "reason": "The source provides related background context without directly confirming or refuting the claim."
    }
