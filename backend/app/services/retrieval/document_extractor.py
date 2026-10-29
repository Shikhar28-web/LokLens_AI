"""
Document Extractor.
Fetches HTML from a given URL and extracts the main article text using Trafilatura.
Includes caching to avoid re-downloading the same article.
"""

import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
import trafilatura

from app.services.retrieval.social_media_extractor import is_social_url, extract_social_post
from app.services.search.cache import _cache  # Reuse the same diskcache instance

logger = logging.getLogger(__name__)

async def fetch_and_extract_article(
    url: str,
    fallback_title: str = "",
    fallback_snippet: str = ""
) -> Optional[dict]:
    """
    Fetch a URL and extract its main article or social post content.
    Returns a dict with 'text', 'title', 'date', 'author', 'is_social' or None if it fails.
    """
    if not url:
        return None

    # Handle social media platforms (X, Instagram, Reddit, YouTube, Telegram)
    if is_social_url(url):
        social_data = await extract_social_post(url, fallback_title, fallback_snippet)
        if social_data and social_data.get("text"):
            return {
                "title": social_data.get("title") or fallback_title,
                "text": social_data.get("text"),
                "author": social_data.get("author", ""),
                "date": social_data.get("date", ""),
                "url": url,
                "is_social": True,
                "platform": social_data.get("platform", "social"),
                "platform_label": social_data.get("platform_label", "Social Media"),
                "retrieved_at": datetime.now(timezone.utc).isoformat()
            }

    # 1. Check cache (use URL hash)
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    cache_key = f"doc:{url_hash}"
    
    cached = _cache.get(cache_key)
    if cached is not None:
        logger.info("Document cache hit for %s", url)
        return cached

    logger.info("Fetching document: %s", url)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }

    try:
        # 2. Download HTML using httpx
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=12.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
            
        if not html:
            logger.warning("Empty HTML returned from %s", url)
            if fallback_snippet:
                return {
                    "title": fallback_title,
                    "text": fallback_snippet,
                    "author": "",
                    "date": "",
                    "url": url,
                    "is_social": False,
                    "retrieved_at": datetime.now(timezone.utc).isoformat()
                }
            return None

        # 3. Extract content using trafilatura
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            include_images=False,
            output_format="json"
        )

        text_content = ""
        doc_title = fallback_title
        doc_author = ""
        doc_date = ""

        if extracted:
            import json
            data = json.loads(extracted)
            text_content = data.get("text", "")
            doc_title = data.get("title", "") or fallback_title
            doc_author = data.get("author", "")
            doc_date = data.get("date", "")

        # Fallback to BeautifulSoup if Trafilatura gave no text
        if not text_content or len(text_content) < 80:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                paragraphs = [p.get_text().strip() for p in soup.find_all("p") if len(p.get_text().strip()) > 30]
                if paragraphs:
                    text_content = " ".join(paragraphs[:10])
                if not doc_title:
                    title_tag = soup.find("title")
                    if title_tag:
                        doc_title = title_tag.get_text().strip()
            except Exception:
                pass

        # Final fallback to search snippet if page blocked or empty
        if not text_content and fallback_snippet:
            text_content = fallback_snippet

        if not text_content:
            logger.warning("Failed to extract content from %s", url)
            return None

        result = {
            "title": doc_title,
            "text": text_content,
            "author": doc_author,
            "date": doc_date,
            "url": url,
            "is_social": False,
            "retrieved_at": datetime.now(timezone.utc).isoformat()
        }

        # 4. Cache it for 24 hours
        _cache.set(cache_key, result, expire=86400)
        return result

    except httpx.HTTPStatusError as e:
        logger.error("HTTP error %s fetching %s", e.response.status_code, url)
    except httpx.RequestError as e:
        logger.error("Network error fetching %s: %s", url, e)
    except Exception as e:
        logger.error("Error extracting document %s: %s", url, e)
        
    # If network error occurred but we have search snippet, use it!
    if fallback_snippet:
        return {
            "title": fallback_title,
            "text": fallback_snippet,
            "author": "",
            "date": "",
            "url": url,
            "is_social": False,
            "retrieved_at": datetime.now(timezone.utc).isoformat()
        }

    return None

