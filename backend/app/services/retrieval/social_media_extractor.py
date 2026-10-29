"""
Social Media Extractor.
Extracts post metadata, text, author, and context from social media URLs
including X (Twitter), Instagram, Reddit, YouTube, Facebook, and Telegram.
Falls back to search snippet when direct scraping is blocked by anti-bot/login walls.
"""

import re
import logging
from typing import Dict, Any, Optional
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup

from app.services.search.cache import _cache

logger = logging.getLogger(__name__)

SOCIAL_DOMAINS = {
    "x.com": "x",
    "twitter.com": "x",
    "mobile.twitter.com": "x",
    "instagram.com": "instagram",
    "www.instagram.com": "instagram",
    "reddit.com": "reddit",
    "www.reddit.com": "reddit",
    "old.reddit.com": "reddit",
    "youtube.com": "youtube",
    "www.youtube.com": "youtube",
    "youtu.be": "youtube",
    "facebook.com": "facebook",
    "www.facebook.com": "facebook",
    "m.facebook.com": "facebook",
    "t.me": "telegram",
    "telegram.me": "telegram",
    "threads.net": "threads",
    "www.threads.net": "threads",
    "linkedin.com": "linkedin",
    "www.linkedin.com": "linkedin",
}

PLATFORM_LABELS = {
    "x": "X (Twitter)",
    "instagram": "Instagram",
    "reddit": "Reddit",
    "youtube": "YouTube",
    "facebook": "Facebook",
    "telegram": "Telegram",
    "threads": "Threads",
    "linkedin": "LinkedIn",
}


def is_social_url(url: str) -> bool:
    """Returns True if the given URL belongs to a known social media platform."""
    if not url:
        return False
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain in SOCIAL_DOMAINS or any(domain.endswith("." + k) for k in SOCIAL_DOMAINS)
    except Exception:
        return False


def get_social_platform(url: str) -> Optional[str]:
    """Returns platform identifier ('x', 'instagram', etc.) or None."""
    if not url:
        return None
    try:
        domain = urlparse(url).netloc.lower()
        for dom, plat in SOCIAL_DOMAINS.items():
            if dom in domain:
                return plat
    except Exception:
        pass
    return None


async def extract_social_post(
    url: str,
    fallback_title: str = "",
    fallback_snippet: str = ""
) -> Dict[str, Any]:
    """
    Extracts structured social media information.
    Attempts HTTP OpenGraph / Twitter Cards extraction, with graceful fallback to search snippets.
    """
    platform = get_social_platform(url) or "social"
    platform_name = PLATFORM_LABELS.get(platform, platform.capitalize())

    cache_key = f"social:{url}"
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    result: Dict[str, Any] = {
        "url": url,
        "platform": platform,
        "platform_label": platform_name,
        "author": "",
        "title": fallback_title,
        "text": fallback_snippet,
        "engagement_summary": "",
        "source_type": "social_media",
        "is_social": True,
        "extracted_via": "snippet_fallback"
    }

    # Attempt to fetch OpenGraph metadata
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
    }

    try:
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=8.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200 and resp.text:
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Extract OpenGraph and Twitter tags
                og_title = (
                    soup.find("meta", property="og:title")
                    or soup.find("meta", attrs={"name": "twitter:title"})
                )
                og_desc = (
                    soup.find("meta", property="og:description")
                    or soup.find("meta", attrs={"name": "twitter:description"})
                    or soup.find("meta", attrs={"name": "description"})
                )
                og_author = (
                    soup.find("meta", attrs={"name": "twitter:creator"})
                    or soup.find("meta", property="article:author")
                )

                extracted_title = og_title.get("content", "").strip() if og_title else ""
                extracted_desc = og_desc.get("content", "").strip() if og_desc else ""
                extracted_author = og_author.get("content", "").strip() if og_author else ""

                # Platform specific parsing
                if platform == "x":
                    # Title often is: "Author on X: 'tweet body...'"
                    if not extracted_author and " on X:" in extracted_title:
                        author_part = extracted_title.split(" on X:")[0].strip()
                        extracted_author = author_part
                    elif not extracted_author and " on Twitter:" in extracted_title:
                        author_part = extracted_title.split(" on Twitter:")[0].strip()
                        extracted_author = author_part
                    
                    # Tweet body is typically in description or title quote
                    tweet_body = extracted_desc or extracted_title

                    # Clean tweet body
                    tweet_body = re.sub(r"^“|”$|^\"|\"$", "", tweet_body).strip()
                    if tweet_body:
                        result["text"] = tweet_body

                elif platform == "instagram":
                    # Instagram og:description often has "X likes, Y comments - Author on Date: 'Caption'"
                    if extracted_desc:
                        match = re.search(r"(\d[\d,]*\s+likes,\s+\d[\d,]*\s+comments)", extracted_desc)
                        if match:
                            result["engagement_summary"] = match.group(1)
                        # Remove engagement prefix if present
                        caption = re.sub(r"^[\d,]+\s+likes,?\s*[\d,]*\s*comments\s*-\s*", "", extracted_desc, flags=re.IGNORECASE)
                        result["text"] = caption.strip() or extracted_desc
                    if not extracted_author and "Instagram: " in extracted_title:
                        extracted_author = extracted_title.split("Instagram: ")[0].replace("on Instagram", "").strip()

                elif platform == "reddit":
                    if extracted_title:
                        result["title"] = extracted_title
                    if extracted_desc and len(extracted_desc) > 20:
                        result["text"] = f"{extracted_title}. {extracted_desc}"

                elif platform == "youtube":
                    if extracted_title:
                        result["title"] = extracted_title
                    if extracted_desc:
                        result["text"] = f"{extracted_title} - {extracted_desc[:400]}"

                if extracted_title and not result["title"]:
                    result["title"] = extracted_title
                if extracted_author:
                    result["author"] = extracted_author
                if extracted_desc and (not result["text"] or len(result["text"]) < len(extracted_desc)):
                    result["text"] = extracted_desc

                result["extracted_via"] = "opengraph_html"

    except Exception as e:
        logger.debug("Live fetch failed for social URL %s (%s), using search snippet.", url, e)

    # Post-process extracted text: ensure fallback text is preserved if empty
    if not result["text"].strip():
        result["text"] = fallback_snippet or fallback_title

    # Extract author from URL path if still empty (e.g., x.com/elonmusk/status/...)
    if not result["author"]:
        try:
            path_parts = [p for p in urlparse(url).path.split("/") if p]
            if platform == "x" and path_parts:
                result["author"] = f"@{path_parts[0]}"
            elif platform == "instagram" and path_parts and path_parts[0] not in ("p", "reel", "stories"):
                result["author"] = f"@{path_parts[0]}"
            elif platform == "reddit" and len(path_parts) >= 2 and path_parts[0] == "r":
                result["author"] = f"r/{path_parts[1]}"
            elif platform == "youtube" and path_parts and path_parts[0].startswith("@"):
                result["author"] = path_parts[0]
        except Exception:
            pass

    # Cache for 12 hours
    _cache.set(cache_key, result, expire=43200)
    return result
