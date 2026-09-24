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
        
    # 1. Neutral query
    queries.append(f'"{claim_text}"') if len(claim_text.split()) < 8 else queries.append(base_query)
    
    # 2. Fact-check query
    queries.append(f"fact check {base_query}")
    
    # 3. Official response
    orgs_or_persons = entities.get("orgs", []) + entities.get("persons", [])
    if orgs_or_persons:
        queries.append(f"{orgs_or_persons[0]} official response statement {base_query}")
    else:
        queries.append(f"official statement response {base_query}")
        
    # 4. Contradictory query
    queries.append(f"{base_query} false fake denied hoax")
    
    # 5. Supporting query
    queries.append(f"{base_query} confirmed proof reported")

    return list(dict.fromkeys(queries))[:5]
