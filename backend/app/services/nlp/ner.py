"""
Named Entity Recognition (NER) module using spaCy.
Extracts entities (PERSON, ORG, GPE, DATE, etc.) from claims.

Phase 14.5 additions:
- extract_spo(): Subject-Predicate-Object extraction via dependency parse
- NER post-processing filters to fix common spaCy misclassifications
"""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Global spacy model loaded lazily to avoid slowing down app startup
_nlp = None

# ── Known Indian cities/states to rescue from wrong spaCy buckets ─────────────
KNOWN_LOCATIONS = {
    "mumbai", "delhi", "bengaluru", "bangalore", "chennai", "hyderabad",
    "kolkata", "pune", "ahmedabad", "surat", "jaipur", "lucknow",
    "kanpur", "nagpur", "patna", "indore", "thane", "bhopal", "visakhapatnam",
    "pimpri", "nashik", "vadodara", "faridabad", "ghaziabad", "ludhiana",
    "agra", "meerut", "rajkot", "varanasi", "srinagar", "amritsar", "allahabad",
    "uttar pradesh", "maharashtra", "gujarat", "rajasthan", "bihar",
    "kerala", "karnataka", "andhra pradesh", "telangana", "punjab",
    "west bengal", "odisha", "assam", "jharkhand", "haryana", "himachal pradesh",
    "india", "pakistan", "china", "usa", "uk", "russia", "ukraine",
    "new delhi", "new york", "washington", "london", "beijing",
}

# ── Generic/collective words wrongly tagged as PERSON by spaCy ────────────────
FALSE_PERSON_TOKENS = {
    "citizens", "people", "voters", "workers", "students", "farmers",
    "protesters", "activists", "officials", "authorities", "residents",
    "journalists", "soldiers", "troops", "police", "government",
    "public", "nation", "everyone", "nobody", "someone", "anyone",
    "children", "women", "men", "youth", "seniors", "migrants",
}

# ── Words commonly misclassified as ORG ───────────────────────────────────────
FALSE_ORG_TOKENS_IF_ALLCAPS_LOCATION = {
    "mumbai", "delhi", "india", "pakistan", "china", "usa", "uk",
}


def get_spacy_model():
    """Lazily load the spaCy model."""
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model 'en_core_web_sm'")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Ensure it is downloaded.")
            raise
    return _nlp


def _is_false_person(text_val: str) -> bool:
    """Return True if this token should NOT be in the persons bucket."""
    return text_val.lower() in FALSE_PERSON_TOKENS


def _is_location_masquerading_as_org(text_val: str) -> bool:
    """Return True if this ORG token is actually a known location."""
    return text_val.lower() in KNOWN_LOCATIONS


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract Named Entities from text.
    Returns a dictionary grouping entities by category:
    - persons: List of PERSON entities (filtered for false positives)
    - orgs: List of ORG (organizations, filtered to remove cities)
    - locations: List of GPE and LOC (geopolitical and locations)
    - dates: List of DATE entities
    - numbers: List of MONEY, PERCENT, QUANTITY, CARDINAL
    - concepts: List of political/domain concepts
    """
    if not text:
        return {"persons": [], "orgs": [], "locations": [], "dates": [], "numbers": [], "concepts": []}

    nlp = get_spacy_model()
    doc = nlp(text)

    entities: Dict[str, set] = {
        "persons": set(),
        "orgs": set(),
        "locations": set(),
        "dates": set(),
        "numbers": set(),
        "concepts": set()
    }

    for ent in doc.ents:
        label = ent.label_
        text_val = ent.text.strip()

        if not text_val:
            continue

        if label == "PERSON":
            # Filter out collective/generic words wrongly tagged as person
            if not _is_false_person(text_val):
                entities["persons"].add(text_val)
            else:
                # Reclaim as concept instead
                entities["concepts"].add(text_val)

        elif label == "ORG":
            # If spaCy tagged a city/country as ORG, move it to locations
            if _is_location_masquerading_as_org(text_val):
                entities["locations"].add(text_val)
            else:
                entities["orgs"].add(text_val)

        elif label in ("GPE", "LOC", "FAC"):
            entities["locations"].add(text_val)

        elif label == "DATE":
            entities["dates"].add(text_val)

        elif label in ("MONEY", "PERCENT", "QUANTITY", "CARDINAL"):
            entities["numbers"].add(text_val)

    # Also check ALL-CAPS single tokens — spaCy often misidentifies acronyms
    # like "MUMBAI" (all caps in a headline) as ORG
    for token in doc:
        token_lower = token.text.lower()
        if token.is_upper and len(token.text) > 2 and token_lower in KNOWN_LOCATIONS:
            # Remove from orgs and persons if wrongly placed, add to locations
            entities["orgs"].discard(token.text)
            entities["persons"].discard(token.text)
            entities["locations"].add(token.text)

    # Noun chunks heuristic to catch political objects and complete numbers
    political_kws = {
        "opposition", "ruling", "party", "voter", "electoral", "roll",
        "revision", "government", "election", "commission", "parliament",
        "assembly", "legislature", "court", "tribunal", "ministry",
    }

    for chunk in doc.noun_chunks:
        c_text = chunk.text.strip()
        c_lower = c_text.lower()

        # Expand numbers like "1.3 crore"
        if any(char.isdigit() for char in c_text) and (
            "crore" in c_lower or "lakh" in c_lower
            or "million" in c_lower or "billion" in c_lower
        ):
            entities["numbers"].add(c_text)

        # Extract political entities / events
        if any(kw in c_lower for kw in political_kws):
            clean_chunk = " ".join([t.text for t in chunk if t.pos_ != "DET"])
            if clean_chunk:
                entities["concepts"].add(clean_chunk)

    return {k: sorted(list(v)) for k, v in entities.items()}


def extract_spo(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract the main Subject-Predicate-Object (SPO) triplet from a sentence
    using spaCy's dependency parse tree.

    Returns:
        Tuple of (subject, predicate, object) — any can be None if not found.

    Examples:
        "IMD issued an Orange Alert for Mumbai"
        → subject="IMD", predicate="issued", object="Orange Alert"
    """
    if not text:
        return None, None, None

    nlp = get_spacy_model()
    doc = nlp(text)

    subject = None
    predicate = None
    obj = None

    # Find the root verb (main predicate)
    root_token = None
    for token in doc:
        if token.dep_ == "ROOT":
            root_token = token
            break

    if root_token is None:
        return None, None, None

    # Predicate = lemma of root verb (or the full verb phrase)
    if root_token.pos_ in ("VERB", "AUX"):
        predicate = root_token.text

    # Find subject (nsubj or nsubjpass)
    for token in root_token.children:
        if token.dep_ in ("nsubj", "nsubjpass"):
            # Include compound parts of subject
            subject_parts = [t.text for t in token.subtree
                             if t.dep_ in ("compound", "nsubj", "nsubjpass", "amod") or t == token]
            subject = " ".join(subject_parts).strip()
            break

    # Find object (dobj, pobj, attr, oprd)
    for token in root_token.children:
        if token.dep_ in ("dobj", "attr", "oprd"):
            obj_parts = [t.text for t in token.subtree
                         if t.dep_ in ("compound", "dobj", "amod", "nummod", "det") or t == token]
            obj = " ".join(obj_parts).strip()
            break

    # Fallback: look for prepositional object if no direct object found
    if obj is None:
        for token in root_token.children:
            if token.dep_ == "prep":
                for child in token.children:
                    if child.dep_ == "pobj":
                        obj_parts = [t.text for t in child.subtree
                                     if t.dep_ in ("compound", "pobj", "amod", "nummod") or t == child]
                        obj = f"{token.text} " + " ".join(obj_parts).strip()
                        break

    return subject or None, predicate or None, obj or None
