"""
Evidence Classifier & Relevance Validation Module.
Evaluates relationship between claims and retrieved web & social media evidence
(SUPPORTS, CONTRADICTS, ATTRIBUTED_CLAIM, SOCIAL_VIRAL_POST, CONTEXT, IRRELEVANT).

v2 improvements:
- Negation detection: negation modifiers within 5 tokens of a keyword flip classification
- Expanded Indian fact-check domain keywords (PIB, AltNews, BoomLive etc.)
- Numeric agreement check with ±5% tolerance for quantities (₹, %, crore, lakh)
- Certified fact-checker domain list bypasses relevance gate
- Tightened relevance overlap threshold (0.18 → 0.22)
"""

from typing import Dict, Any
import logging
import re

logger = logging.getLogger(__name__)

# ── Negation context window (tokens to look behind/ahead of a keyword) ────────
_NEGATION_WORDS = {"not", "never", "no", "neither", "nor", "without", "false",
                   "deny", "denied", "refute", "refuted", "wrong", "incorrect",
                   "n't", "cannot", "cant"}

# ── Certified Indian fact-checker & government authority domains ──────────────
CERTIFIED_FACTCHECK_DOMAINS = {
    "altnews.in", "boomlive.in", "factly.in", "vishvasnews.com",
    "factchecker.in", "thequint.com", "newschecker.in", "indiacheck.org",
    "pib.gov.in", "pib.nic.in", "mygov.in", "government.in",
    "snopes.com", "politifact.com", "fullfact.org"
}

CONTRADICTION_KEYWORDS = [
    # --- existing ---
    "false", "fake", "deny", "denies", "denied", "debunk", "debunked",
    "hoax", "fabricated", "doctored", "morphed", "misleading", "fact check",
    "no such", "untrue", "busted", "misinformation", "disinformation",
    "refuted", "baseless", "incorrect", "distortion", "claims falsely",
    "not true", "never said", "clarified that no", "distorted",
    # --- Indian fact-check specific additions ---
    "claim is false", "viral post misleading", "no such order", "pib fact check",
    "government clarification", "news is fake", "morphed image", "old video",
    "out of context", "altered image", "edited photo", "misleading post",
    "propaganda", "rumour", "rumor", "unverified claim", "no evidence",
    "not confirmed", "contradicts", "misleads", "wrong claim",
    "this is false", "this is fake", "fact check reveals",
    "misleading claim", "misleading information", "does not show",
    "video is old", "image is old", "photo is old", "taken out of context",
    "pib clarification", "ministry clarification", "government denies",
    "official denial", "no order issued", "no notification issued"
]

SUPPORT_KEYWORDS = [
    # --- existing ---
    "confirmed", "announced", "approved", "passed", "issued", "held",
    "inaugurated", "declared", "verified", "signed", "stated", "enacted",
    "agreed", "launched", "according to official records", "data shows",
    # --- official/government additions ---
    "officially confirmed", "government notified", "gazette notification",
    "ministry statement", "press release", "official statement",
    "pib release", "government announced", "officially launched",
    "government approved", "cabinet approved", "lok sabha passed",
    "rajya sabha passed", "presidential assent", "governor signed",
    "supreme court ruled", "high court ruled", "central government",
    "state government announced", "official data", "census data",
    "rbi notification", "sebi circular", "government order",
    "government circular", "official notification", "statutory order",
    "government report", "ministry report", "niti aayog", "is true",
    "this is true", "fact: true", "verdict: true", "status: true"
]

ATTRIBUTION_KEYWORDS = [
    "claimed", "alleged", "accused", "argued", "said", "posted", "tweeted",
    "shared", "captioned", "circulating", "viral video shows", "purportedly"
]


# ── Helper: negation detection ────────────────────────────────────────────────
def _has_negated_keyword(text: str, keywords: list, window: int = 5) -> bool:
    """
    Returns True ONLY if a keyword appears AND is preceded or followed by a
    negation word within `window` tokens. Used to detect "not confirmed",
    "never said it was false" etc. to avoid flipping the wrong classification.
    """
    tokens = re.findall(r"[\w']+", text.lower())
    for i, tok in enumerate(tokens):
        for kw in keywords:
            kw_tokens = kw.split()
            # single-token keyword fast path
            if len(kw_tokens) == 1 and tok == kw:
                start = max(0, i - window)
                end = min(len(tokens), i + window + 1)
                context = set(tokens[start:end])
                if context & _NEGATION_WORDS:
                    return True
    return False


def _keyword_present_without_negation(text: str, keywords: list, window: int = 5) -> tuple:
    """
    Returns (found: bool, matched_keywords: list).
    A keyword match is counted only when NOT negated within `window` tokens.
    Multi-word phrases are matched via substring; single-word via token list.
    """
    t_lower = text.lower()
    tokens = re.findall(r"[\w']+", t_lower)
    matched = []
    for kw in keywords:
        kw_tokens = kw.lower().split()
        if len(kw_tokens) == 1:
            # token-level search with negation window check
            for i, tok in enumerate(tokens):
                if tok == kw_tokens[0]:
                    start = max(0, i - window)
                    end = min(len(tokens), i + window + 1)
                    context = set(tokens[start:end]) - {tok}
                    if not (context & _NEGATION_WORDS):
                        matched.append(kw)
                    break
        else:
            # phrase-level substring match
            if kw.lower() in t_lower:
                # simple negation check: look 5 words before the phrase
                idx = t_lower.find(kw.lower())
                pre_text = t_lower[max(0, idx - 60): idx]
                pre_toks = set(re.findall(r"[\w']+", pre_text))
                if not (pre_toks & _NEGATION_WORDS):
                    matched.append(kw)
    return bool(matched), matched


# ── Helper: numeric agreement check ──────────────────────────────────────────
def _numbers_agree(claim: str, evidence: str, tolerance: float = 0.05) -> bool:
    """
    Returns True if at least one number in the claim appears in the evidence
    within ±tolerance (5% by default). Handles crore/lakh multipliers.
    """
    def _to_float(s: str):
        s = s.replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None

    def _extract_numbers(text: str):
        # Match numeric values optionally followed by crore/lakh/million/billion
        pattern = r'[\d,]+(?:\.\d+)?(?:\s*(?:crore|lakh|million|billion|cr|mn|bn))?'
        raw = re.findall(pattern, text.lower())
        results = []
        for r in raw:
            multiplier = 1
            if "crore" in r or " cr" in r:
                multiplier = 1e7
            elif "lakh" in r:
                multiplier = 1e5
            elif "million" in r or " mn" in r:
                multiplier = 1e6
            elif "billion" in r or " bn" in r:
                multiplier = 1e9
            numeric_part = re.search(r'[\d,]+(?:\.\d+)?', r)
            if numeric_part:
                val = _to_float(numeric_part.group())
                if val is not None:
                    results.append(val * multiplier)
        return results

    claim_nums = _extract_numbers(claim)
    if not claim_nums:
        return False
    ev_nums = _extract_numbers(evidence)
    if not ev_nums:
        return False
    for cn in claim_nums:
        for en in ev_nums:
            if cn == 0:
                if en == 0:
                    return True
                continue
            if abs(cn - en) / abs(cn) <= tolerance:
                return True
    return False


def _get_domain(url: str) -> str:
    """Extract bare domain from a URL string."""
    m = re.search(r'(?:https?://)?(?:www\.)?([^/?\s]+)', url or "")
    return m.group(1).lower() if m else ""


def check_source_relevance(claim: str, document_text: str, is_social: bool = False,
                           source_url: str = "") -> dict:
    """
    Source relevance and validation layer v2.
    - Certified fact-checker domains always pass (bypass overlap gate)
    - Tightened overlap threshold: 0.18 → 0.22
    - Social media flexible matching unchanged
    """
    if not claim or not document_text:
        return {"relevant": False, "score": 0.0, "reason": "Empty claim or document."}

    c_lower = claim.lower()
    d_lower = document_text.lower()

    # Fast-pass: certified fact-checker or government domains always admitted
    domain = _get_domain(source_url)
    if domain and any(domain.endswith(d) for d in CERTIFIED_FACTCHECK_DOMAINS):
        return {"relevant": True, "score": 0.95, "reason": f"Certified fact-check/authority domain: {domain}"}

    # Social media post — flexible matching
    if is_social:
        if len(d_lower.strip()) < 20:
            return {"relevant": False, "score": 0.1, "reason": "Social media post is too short to evaluate."}
        c_keywords = [w for w in re.findall(r'[\w\u0900-\u097F]+', c_lower) if len(w) > 3]
        overlap_social = sum(1 for w in c_keywords if w in d_lower)
        if overlap_social >= 1 or len(c_keywords) <= 2:
            return {"relevant": True, "score": 0.75, "reason": "Social media post matches core claim subject."}

    # Strict check for standard web articles
    from app.services.nlp.ner import extract_entities
    claim_ents = extract_entities(claim)

    # Check key ORGs — reject only if ALL named orgs are missing
    missing_orgs = []
    for org in claim_ents.get("orgs", []):
        org_words = [w for w in org.lower().split() if len(w) > 3]
        if org_words and not any(w in d_lower for w in org_words):
            missing_orgs.append(org)

    if missing_orgs and len(missing_orgs) == len(claim_ents.get("orgs", [])) and len(claim_ents.get("orgs", [])) > 1:
        return {"relevant": False, "score": 0.0, "reason": f"Source is missing key organizations: {missing_orgs}"}

    # Keyword overlap — tightened threshold 0.22 (was 0.18)
    c_words = set(w for w in re.findall(r'[\w\u0900-\u097F]+', c_lower) if len(w) > 3)
    d_words = set(re.findall(r'[\w\u0900-\u097F]+', d_lower))

    overlap = len(c_words & d_words)
    ratio = overlap / len(c_words) if c_words else 1.0

    if ratio < 0.22 and overlap < 2:
        return {"relevant": False, "score": ratio, "reason": "Source has insufficient keyword overlap with the claim."}

    return {"relevant": True, "score": min(ratio + 0.3, 1.0), "reason": "Source matches claim entities and topics."}


def classify_evidence(claim: str, evidence_text: str, is_social: bool = False) -> Dict[str, Any]:
    """
    Classifies relationship between claim and evidence v2:
    - SUPPORTS
    - CONTRADICTS
    - ATTRIBUTED_CLAIM
    - SOCIAL_VIRAL_POST
    - CONTEXT
    - IRRELEVANT

    Improvements:
    - Negation-aware keyword matching (flips classification when keyword is negated)
    - Numeric agreement check with ±5% tolerance
    - Expanded keyword lists for Indian fact-check context
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

    # ── Negation-aware contradiction / support detection ──────────────────────
    has_contradiction, contra_found = _keyword_present_without_negation(e_lower, CONTRADICTION_KEYWORDS)
    has_support, support_found = _keyword_present_without_negation(e_lower, SUPPORT_KEYWORDS)

    # If a CONTRADICTION keyword appears but is itself negated
    # (e.g. "the news is not false"), that is actually evidence of SUPPORT
    negated_contradiction = _has_negated_keyword(e_lower, CONTRADICTION_KEYWORDS)
    negated_support = _has_negated_keyword(e_lower, SUPPORT_KEYWORDS)

    # Flip: negated contradiction → treat as support signal
    if negated_contradiction and not has_contradiction:
        has_support = True
        support_found.append("negated_contradiction")
    # Flip: negated support → treat as contradiction signal
    if negated_support and not has_support:
        has_contradiction = True
        contra_found.append("negated_support")

    # Keyword overlap count
    c_words = [w for w in re.findall(r'[\w\u0900-\u097F]+', c_lower) if len(w) > 3]
    d_words = set(re.findall(r'[\w\u0900-\u097F]+', e_lower))
    overlap_count = sum(1 for w in c_words if w in d_words)

    # Numeric agreement
    num_agree = _numbers_agree(c_lower, e_lower)

    # ── 1. Check for Irrelevant ───────────────────────────────────────────────
    if overlap_count < 1 and not is_social:
        return {
            "relationship": "IRRELEVANT",
            "is_allegation": False,
            "confidence": 0.85,
            "reason": "The text does not directly address the specifics of the claim."
        }

    # ── 2. Strong Contradiction ───────────────────────────────────────────────
    if has_contradiction:
        found_kw = contra_found[:2]
        return {
            "relationship": "CONTRADICTS",
            "is_allegation": is_allegation,
            "confidence": 0.90,
            "reason": f"The source explicitly refutes the claim using terms such as '{', '.join(found_kw)}'."
        }

    # ── 3. Social Media Viral Post ────────────────────────────────────────────
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

    # ── 4. Attributed Claim (someone said it, not verified fact) ─────────────
    if is_allegation and not has_support:
        return {
            "relationship": "ATTRIBUTED_CLAIM",
            "is_allegation": True,
            "confidence": 0.82,
            "reason": "The source reports that an entity made this allegation, but does not independently establish it as fact."
        }

    # ── 5. Strong Support ─────────────────────────────────────────────────────
    # Triggers on: support keyword, high overlap, numeric agreement, or ₹ symbol match
    if has_support or overlap_count >= 3 or num_agree or ("₹" in c_lower and "₹" in e_lower):
        found_kw = support_found[:2] if support_found else []
        reason_parts = []
        if found_kw:
            reason_parts.append(f"keywords: {', '.join(found_kw)}")
        if num_agree:
            reason_parts.append("numeric values agree within ±5%")
        if overlap_count >= 3:
            reason_parts.append(f"{overlap_count} claim keywords matched")
        return {
            "relationship": "SUPPORTS",
            "is_allegation": False,
            "confidence": 0.88,
            "reason": "The source explicitly reports verified facts that align with the claim" +
                      (f" ({'; '.join(reason_parts)})." if reason_parts else ".")
        }

    # ── 6. General Context ────────────────────────────────────────────────────
    res = {
        "relationship": "CONTEXT",
        "is_allegation": is_allegation,
        "confidence": 0.60,
        "reason": "The source provides related background context without directly confirming or refuting the claim."
    }

    # ── 7. FEVER ML Claim-Evidence Verification Signal ───────────────────────
    try:
        from ml.inference.fever_inference import predict as predict_fever
        fever_pred = predict_fever(claim, evidence)
        if fever_pred.get("processing_status") == "ok":
            res["fever_nli"] = fever_pred
            f_label = fever_pred.get("label")
            f_conf = fever_pred.get("confidence", 0.0)
            if f_label == "SUPPORTS" and f_conf >= 0.70:
                res["relationship"] = "SUPPORTS"
                res["confidence"] = round(f_conf, 2)
                res["reason"] = f"FEVER ML model verified claim as supported by retrieved evidence (confidence: {f_conf:.2f})."
            elif f_label == "REFUTES" and f_conf >= 0.70:
                res["relationship"] = "CONTRADICTS"
                res["confidence"] = round(f_conf, 2)
                res["reason"] = f"FEVER ML model verified claim as contradicted by retrieved evidence (confidence: {f_conf:.2f})."
    except Exception as e:
        logger.debug("FEVER NLI inference in evidence classifier skipped: %s", e)

    return res
