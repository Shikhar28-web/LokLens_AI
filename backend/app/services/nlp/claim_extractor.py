"""
Claim Extractor module.
Splits paragraphs into atomic sentences/claims and filters out non-claims.
"""

from typing import List
from app.services.nlp.ner import get_spacy_model

def extract_claims(text: str) -> List[str]:
    """
    Splits text into sentences and filters them to find verifiable claims.
    A simple heuristic for Phase 2:
    - Must have a subject and a verb.
    - Must be longer than a few words.
    - Shouldn't be a question.
    """
    if not text:
        return []

    nlp = get_spacy_model()
    doc = nlp(text)

    import re
    claims = []
    
    # Delimiters to split compound claims
    # We only split on comma+conjunction, EXCEPT when followed by dependent clause markers
    delimiters = r', and (?!therefore|because|so|thus|hence)|, but |, while |, which '
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text or len(sent_text.split()) < 4 or sent_text.endswith("?"):
            continue
            
        # Attempt to split into atomic parts
        atomic_parts = re.split(delimiters, sent_text, flags=re.IGNORECASE)
        
        for part in atomic_parts:
            part = part.strip()
            if len(part.split()) < 4:
                continue
                
            part_doc = nlp(part)
            has_verb = any(t.pos_ == "VERB" for t in part_doc)
            has_nsubj = any(t.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass") for t in part_doc)
            
            if has_verb or has_nsubj: # Relaxed heuristic to catch atomic parts
                claims.append(part)
            
    # If heuristics filtered everything but there is text, fallback to returning the whole text as one claim
    # (Useful for short inputs where parsing might fail)
    if not claims and len(text.split()) >= 3 and not text.endswith("?"):
        claims.append(text.strip())

    return claims
