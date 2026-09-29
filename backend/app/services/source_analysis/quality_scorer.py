"""Source Quality Scorer module."""
from urllib.parse import urlparse

def score_source_quality(url: str) -> dict:
    """
    Scores the quality of a source based on its domain.
    Returns a dict with authority_score and source_type.
    """
    if not url:
        return {"authority_score": 0.1, "source_type": "unknown"}
        
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
    except Exception:
        return {"authority_score": 0.1, "source_type": "unknown"}
        
    # High authority
    if domain.endswith(".gov") or domain.endswith(".gov.in") or domain.endswith(".edu"):
        return {"authority_score": 1.0, "source_type": "government"}
        
    # Known news (just a few for testing)
    known_news = [
        "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com", "thehindu.com", 
        "indianexpress.com", "timesofindia.indiatimes.com", "economictimes.indiatimes.com",
        "ndtv.com", "indiatoday.in"
    ]
    
    if any(n in domain for n in known_news):
        return {"authority_score": 0.8, "source_type": "news"}
        
    # Known social
    known_social = ["twitter.com", "x.com", "facebook.com", "reddit.com", "instagram.com"]
    if any(s in domain for s in known_social):
        return {"authority_score": 0.3, "source_type": "social"}
        
    # Default generic
    return {"authority_score": 0.5, "source_type": "unknown"}
