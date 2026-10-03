"""
Claim Extractor module.
Extracts factual claims using an LLM, falling back to deterministic extraction.
"""

from typing import List, Dict, Any
from app.services.nlp.ner import get_spacy_model
from app.services.nlp.llm_client import call_llm_json
import logging
import json

logger = logging.getLogger(__name__)

def extract_structured_claims_llm(text: str) -> List[Dict[str, Any]]:
    """Uses a single LLM call to extract atomic claims and their search queries."""
    if not text:
        return []
        
    prompt = f"""
You are a fact-checking assistant. Process the following text.
1. Break the entire text down into small, individual sentences or logical parts. Every single sentence must be treated as a separate claim.
2. Extract the exact text for each part. Do not invent or reword.
3. Every search query MUST strictly be the extracted sentence itself or a direct substring of it. DO NOT append ANY external words, metadata, or context.
4. Do NOT include OCR garbage or UI labels.

Text:
{text}

Output JSON format exactly:
{{
  "claims": [
    {{
      "claim_text": "First small sentence exactly as it appears.",
      "entities": ["entity1", "entity2"],
      "numbers": ["150", "60"],
      "search_queries": ["exact claim substring"]
    }},
    {{
      "claim_text": "Second small sentence exactly as it appears.",
      "entities": ["entity3"],
      "numbers": [],
      "search_queries": ["exact claim substring"]
    }}
  ]
}}
"""
    result = call_llm_json(prompt)
    if result:
        return result.get("claims", [])
        
    return []

def extract_claims(text: str) -> List[str]:
    """Fallback deterministic claim extractor if LLM fails/is unavailable."""
    if not text:
        return []

    nlp = get_spacy_model()
    doc = nlp(text)
    import re
    claims = []
    delimiters = r', and (?!therefore|because|so|thus|hence)|, but |, while |, which '
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text or len(sent_text.split()) < 4 or sent_text.endswith("?"):
            continue
        atomic_parts = re.split(delimiters, sent_text, flags=re.IGNORECASE)
        for part in atomic_parts:
            part = part.strip()
            if len(part.split()) < 4: continue
            part_doc = nlp(part)
            has_verb = any(t.pos_ == "VERB" for t in part_doc)
            has_nsubj = any(t.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass") for t in part_doc)
            if has_verb or has_nsubj:
                claims.append(part)
    
    if not claims and len(text.split()) >= 3 and not text.endswith("?"):
        claims.append(text.strip())

    return claims
