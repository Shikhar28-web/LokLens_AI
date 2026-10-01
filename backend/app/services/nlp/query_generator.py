"""
Query Generator module.
Generates search engine queries from a claim and its entities.
"""

from typing import Dict, List
from app.services.nlp.tokenizer import extract_keywords

def generate_queries(claim_text: str, entities: Dict[str, List[str]]) -> List[str]:
    """
    Generate multiple search queries for a given claim.
    Creates variations to maximize retrieval recall.
    """
    if not claim_text:
        return []

    queries = []
    
    # Base entities
    core_terms = []
    for cat in ["orgs", "persons", "locations", "concepts", "numbers"]:
        core_terms.extend(entities.get(cat, []))
        
    keywords = extract_keywords(claim_text)
    
    # Deduplicate terms
    primary_terms = []
    seen_lower = set()
    for term in core_terms + keywords:
        lower_term = term.lower()
        if lower_term not in seen_lower:
            seen_lower.add(lower_term)
            primary_terms.append(term)
            
    base_query = " ".join(primary_terms[:6])
    if not base_query:
        base_query = claim_text
        
    # 1. Neutral/Exact query (if short enough)
    if len(claim_text.split()) < 8:
        queries.append(f'"{claim_text}"')
    
    # 2. Broad keyword query
    queries.append(base_query)
    
    # 3. Entity-focused query (who and where)
    orgs_or_persons = entities.get("orgs", []) + entities.get("persons", [])
    locations = entities.get("locations", [])
    if orgs_or_persons or locations:
        entity_query = " ".join(orgs_or_persons + locations)
        queries.append(f"{entity_query} fact check")
        
    # 4. Contextual keyword query
    if len(primary_terms) > 3:
        queries.append(" ".join(primary_terms[1:5]))
        
    # 5. News/Report query
    queries.append(f"{primary_terms[0] if primary_terms else base_query} reported news")

    return list(dict.fromkeys(queries))[:5]
