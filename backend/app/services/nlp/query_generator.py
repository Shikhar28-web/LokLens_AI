"""
Query Generator module.
Generates search engine queries from a claim and its entities.
"""

from typing import Dict, List
from app.services.nlp.tokenizer import extract_keywords

def generate_queries(claim_text: str, entities: Dict[str, List[str]], global_context: str = "") -> List[str]:
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
        
    prefix = f"{global_context} " if global_context else ""
        
    # 1. Neutral/Exact query (if short enough)
    if len(claim_text.split()) < 8:
        queries.append(f'{prefix}"{claim_text}"'.strip())
    
    # 2. Broad keyword query
    queries.append(f"{prefix}{base_query}".strip())
    
    # 3. Entity-focused query (who and where)
    orgs_or_persons = entities.get("orgs", []) + entities.get("persons", [])
    locations = entities.get("locations", [])
    if orgs_or_persons or locations:
        entity_query = " ".join(orgs_or_persons + locations)
        queries.append(f"{prefix}{entity_query} fact check".strip())
        
    # 4. Contextual keyword query
    if len(primary_terms) > 3:
        queries.append(f"{prefix}{' '.join(primary_terms[1:5])}".strip())
        
    # 5. News/Report query
    queries.append(f"{prefix}{primary_terms[0] if primary_terms else base_query} reported news".strip())
    
    # 6. Official sources queries (RBI, NPCI, ECI, IMD, BMC)
    claim_lower = claim_text.lower()
    if any(x in claim_lower for x in ["election", "voter", "poll", "eci"]):
        queries.append(f"{prefix}{base_query} site:eci.gov.in".strip())
    if any(x in claim_lower for x in ["rbi", "bank", "currency", "rupee", "note"]):
        queries.append(f"{prefix}{base_query} site:rbi.org.in".strip())
    if any(x in claim_lower for x in ["upi", "npci", "payment", "transaction"]):
        queries.append(f"{prefix}{base_query} site:npci.org.in".strip())
    if any(x in claim_lower for x in ["weather", "rain", "alert", "cyclone", "imd"]):
        queries.append(f"{prefix}{base_query} site:mausam.imd.gov.in".strip())
    if any(x in claim_lower for x in ["mumbai", "bmc", "civic"]):
        queries.append(f"{prefix}{base_query} site:mcgm.gov.in".strip())

    return list(dict.fromkeys(queries))[:5]
