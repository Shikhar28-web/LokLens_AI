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
    
    # 1. The exact claim text as a natural sentence
    queries.append(claim_text)
    
    # 2. Quoted exact search if the claim is concise
    if len(claim_text.split()) < 10:
        queries.append(f'"{claim_text}"')
        
    # 3. Meaningful NLP reduction (claim text minus stopwords, keeping order)
    from app.services.nlp.ner import get_spacy_model
    nlp = get_spacy_model()
    doc = nlp(claim_text)
    
    meaningful_words = []
    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            meaningful_words.append(token.text)
            
    reduced_query = " ".join(meaningful_words)
    if reduced_query and reduced_query.lower() != claim_text.lower():
        queries.append(reduced_query)

    # 4. Entity-focused natural query (Who and Where)
    orgs_or_persons = entities.get("orgs", []) + entities.get("persons", [])
    locations = entities.get("locations", [])
    if orgs_or_persons and locations:
        entity_query = f"{' '.join(orgs_or_persons)} in {' '.join(locations)}"
        if entity_query.lower() not in [q.lower() for q in queries]:
            queries.append(entity_query)

    # Final deduplication
    final_queries = []
    seen_q = set()
    for q in queries:
        if q.lower() not in seen_q:
            seen_q.add(q.lower())
            final_queries.append(q)

    return final_queries[:3]
