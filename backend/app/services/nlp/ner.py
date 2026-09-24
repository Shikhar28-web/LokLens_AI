"""
Named Entity Recognition (NER) module using spaCy.
Extracts entities (PERSON, ORG, GPE, DATE, etc.) from claims.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Global spacy model loaded lazily to avoid slowing down app startup
_nlp = None

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

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract Named Entities from text.
    Returns a dictionary grouping entities by category:
    - persons: List of PERSON entities
    - orgs: List of ORG (organizations)
    - locations: List of GPE and LOC (geopolitical and locations)
    - dates: List of DATE entities
    - numbers: List of MONEY, PERCENT, QUANTITY, CARDINAL
    """
    if not text:
        return {"persons": [], "orgs": [], "locations": [], "dates": [], "numbers": []}

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
            entities["persons"].add(text_val)
        elif label == "ORG":
            entities["orgs"].add(text_val)
        elif label in ("GPE", "LOC", "FAC"):
            entities["locations"].add(text_val)
        elif label == "DATE":
            entities["dates"].add(text_val)
        elif label in ("MONEY", "PERCENT", "QUANTITY", "CARDINAL"):
            entities["numbers"].add(text_val)

    # Noun chunks heuristic to catch political objects and complete numbers
    political_kws = {"opposition", "ruling", "party", "voter", "electoral", "roll", "revision", "government", "election", "commission"}
    
    for chunk in doc.noun_chunks:
        c_text = chunk.text.strip()
        c_lower = c_text.lower()
        
        # Expand numbers like "1.3 crore"
        if any(char.isdigit() for char in c_text) and ("crore" in c_lower or "lakh" in c_lower or "million" in c_lower or "billion" in c_lower):
            entities["numbers"].add(c_text)
            
        # Extract political entities / events
        if any(kw in c_lower for kw in political_kws):
            # Clean up determiners (e.g., "the opposition" -> "opposition")
            clean_chunk = ' '.join([t.text for t in chunk if t.pos_ != "DET"])
            if clean_chunk:
                entities["concepts"].add(clean_chunk)

    return {k: sorted(list(v)) for k, v in entities.items()}
