"""
Query Generator module.
Generates search engine queries from a claim and its entities.
"""

from typing import Dict, List
from app.services.nlp.tokenizer import extract_keywords

def generate_queries(claim_text: str, entities: Dict[str, List[str]], global_context: str = "") -> List[str]:
    """
    Generate multiple search queries for a given claim.
    Maintains the natural sentence structure instead of creating word salad.
    """
    if not claim_text:
        return []

    queries = []
    
    # 1. Quoted exact search if the claim is concise and clean
    # Only if it's less than 8 words and doesn't have weird characters
    import re
    if len(claim_text.split()) <= 8 and not re.search(r'[^\w\s]', claim_text):
        queries.append(f'"{claim_text}"')
        
    # 2. Meaningful NLP reduction (claim text minus stopwords, keeping order, max 8 words)
    from app.services.nlp.ner import get_spacy_model
    nlp = get_spacy_model()
    doc = nlp(claim_text)
    
    meaningful_words = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space and token.text.isalnum():
            meaningful_words.append(token.text)
            
    reduced_query = " ".join(meaningful_words[:8]) # Strict 8 word limit
    if reduced_query:
        queries.append(reduced_query)

    # 3. Entity-focused natural query (Who + Where + Concept)
    orgs_or_persons = entities.get("orgs", []) + entities.get("persons", [])
    locations = entities.get("locations", [])
    concepts = entities.get("concepts", [])
    
    entity_parts = []
    if orgs_or_persons: entity_parts.extend(orgs_or_persons[:2])
    if locations: entity_parts.extend(locations[:1])
    if concepts: entity_parts.extend(concepts[:1])
    
    if entity_parts:
        entity_query = " ".join(entity_parts)[:60] # Limit character length
        if entity_query.lower() not in [q.lower() for q in queries]:
            queries.append(entity_query)
            
    # Fallback to a slice of the raw text if nothing else works
    if not queries:
        queries.append(" ".join(claim_text.split()[:5]))

    # Final deduplication
    final_queries = []
    seen_q = set()
    for q in queries:
        if q.lower() not in seen_q:
            seen_q.add(q.lower())
            final_queries.append(q)

    return final_queries[:3]
