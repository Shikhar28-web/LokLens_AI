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
        
    # Fact-checking organizations
    fact_checkers = [
        "altnews.in", "boomlive.in", "factly.in", "snopes.com", "politifact.com",
        "factcheck.org", "pib.gov.in", "vishvasnews.com", "thip.media", "newschecker.in"
    ]
    if any(fc in domain for fc in fact_checkers):
        return {"authority_score": 0.95, "source_type": "fact_checker", "platform": "fact_check"}

    # High authority government & academic
    if domain.endswith(".gov") or domain.endswith(".gov.in") or domain.endswith(".nic.in") or domain.endswith(".edu"):
        return {"authority_score": 1.0, "source_type": "government",   "platform": "government"}
        
    # Known mainstream news
    known_news = [
        "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com", "thehindu.com", 
        "indianexpress.com", "timesofindia.indiatimes.com", "economictimes.indiatimes.com",
        "ndtv.com", "indiatoday.in", "hindustantimes.com", "theprint.in",
        "scroll.in", "thewire.in", "livemint.com", "aninews.in", "ptinews.com",
        "aljazeera.com", "nytimes.com", "washingtonpost.com", "theguardian.com"
    ]
    if any(n in domain for n in known_news):
        return {"authority_score": 0.85, "source_type": "news", "platform": "news"}
        
    # Social media platforms
    social_map = {
        "x.com": "x",
        "twitter.com": "x",
        "instagram.com": "instagram",
        "reddit.com": "reddit",
        "youtube.com": "youtube",
        "youtu.be": "youtube",
        "threads.net": "threads",
        "facebook.com": "facebook",
        "t.me": "telegram"
    }
    for s_dom, s_plat in social_map.items():
        if s_dom in domain:
            return {
                "authority_score": 0.45,
                "source_type": "social_media",
                "platform": s_plat,
                "is_social": True
            }
        
    # Default generic web source
    return {"authority_score": 0.55, "source_type": "web", "platform": "web"}

