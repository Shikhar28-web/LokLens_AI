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

from app.services.search.cache import _cache  # Reuse the same diskcache instance

logger = logging.getLogger(__name__)

async def fetch_and_extract_article(url: str) -> Optional[dict]:
    """
    Fetch a URL and extract its main article content.
    Returns a dict with 'text', 'title', 'date', 'author' or None if it fails.
    """
    # 1. Check cache (use URL hash)
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    cache_key = f"doc:{url_hash}"
    
    cached = _cache.get(cache_key)
    if cached is not None:
        logger.info("Document cache hit for %s", url)
        return cached

    logger.info("Fetching document: %s", url)
    headers = {
        "User-Agent": "LokLensBot/1.0 (https://loklens.ai; contact@loklens.ai) httpx/0.27+"
    }

    try:
        # 2. Download HTML using httpx
        async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=15.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
            
        if not html:
            logger.warning("Empty HTML returned from %s", url)
            return None
        # 3. Extract content using trafilatura
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            include_images=False,
            output_format="json"
        )

        if not extracted:
            logger.warning("Trafilatura failed to extract content from %s", url)
            return None

        # trafilatura's JSON output is a string, parse it
        import json
        data = json.loads(extracted)
        
        result = {
            "title": data.get("title", ""),
            "text": data.get("text", ""),
            "author": data.get("author", ""),
            "date": data.get("date", ""),
            "url": url,
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
        
    return None
