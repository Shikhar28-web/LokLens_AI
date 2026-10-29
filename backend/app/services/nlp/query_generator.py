"""
Query Generator module.
Generates multi-perspective search engine queries from a claim and its entities,
including direct factual searches, fact-check targeted searches, and social media scoped searches (X, Instagram, Reddit).
"""

import re
from typing import Dict, List, Any


def generate_queries(claim_text: str, entities: Dict[str, List[str]] = None, global_context: str = "") -> List[str]:
    """
    Generate diverse search queries for a given claim:
    1. Core concise claim query (natural phrase)
    2. Fact-check specific query ('<keywords> fact check')
    3. Social media search on X/Twitter ('site:x.com OR site:twitter.com <keywords>')
    4. Social media search on Instagram/Reddit ('site:instagram.com OR site:reddit.com <keywords>')
    5. Entity & Temporal query (Who + What + Year)
    """
    if not claim_text or not claim_text.strip():
        return []

    entities = entities or {}
    queries = []
    
    clean_text = claim_text.strip()

    # 1. Quoted exact search if the claim is concise (4 to 8 words)
    words = clean_text.split()
    if 4 <= len(words) <= 7 and not re.search(r'[^\w\s\u0900-\u097F]', clean_text):
        queries.append(f'"{clean_text}"')

    # 2. Extract meaningful keywords (handling both English and Devanagari)
    from app.services.nlp.ner import get_spacy_model
    nlp = get_spacy_model()
    doc = nlp(clean_text)
    
    meaningful_words = []
    for token in doc:
        # Check non-stopword, non-punctuation
        if not token.is_stop and not token.is_punct and not token.is_space and len(token.text) > 1:
            meaningful_words.append(token.text)
            
    reduced_core = " ".join(meaningful_words[:7])
    if reduced_core:
        queries.append(reduced_core)
    else:
        reduced_core = " ".join(words[:6])
        queries.append(reduced_core)

    # 3. Entity-focused query
    orgs_or_persons = entities.get("orgs", []) + entities.get("persons", [])
    locations = entities.get("locations", [])
    concepts = entities.get("concepts", [])
    
    entity_parts = []
    if orgs_or_persons:
        entity_parts.extend(orgs_or_persons[:2])
    if locations:
        entity_parts.extend(locations[:1])
    if concepts:
        entity_parts.extend(concepts[:1])
        
    if entity_parts:
        ent_query = " ".join(entity_parts)[:60]
        if ent_query.lower() not in [q.lower() for q in queries]:
            queries.append(ent_query)

    # 4. Fact-check targeted query
    core_kw = " ".join(meaningful_words[:5]) if meaningful_words else " ".join(words[:4])
    fact_check_query = f"{core_kw} fact check"
    queries.append(fact_check_query)

    # 5. Social Media Query A: X / Twitter
    # e.g., site:x.com OR site:twitter.com <keywords>
    if core_kw:
        x_query = f"(site:x.com OR site:twitter.com) {core_kw}"
        queries.append(x_query)

    # 6. Social Media Query B: Instagram / Reddit
    if core_kw:
        insta_reddit_query = f"(site:instagram.com OR site:reddit.com) {core_kw}"
        queries.append(insta_reddit_query)

    # 7. Temporal query if year or date detected
    year_match = re.search(r'\b(19\d\d|20[0-3]\d)\b', clean_text)
    if year_match:
        year_str = year_match.group(1)
        temporal_query = f"{core_kw} {year_str}"
        queries.append(temporal_query)

    # Final deduplication while preserving order
    final_queries = []
    seen = set()
    for q in queries:
        q_norm = q.strip().lower()
        if q_norm and q_norm not in seen:
            seen.add(q_norm)
            final_queries.append(q.strip())

    return final_queries[:5]  # Cap at 5 targeted queries (covers web, fact-checks, and social media)
