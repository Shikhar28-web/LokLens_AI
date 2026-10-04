"""
DuckDuckGo Search Provider.
Uses httpx and BeautifulSoup to scrape lite.duckduckgo.com.
Free, no API key required, highly rate-limited so caching is strictly used.
"""

import asyncio
import logging
from typing import Any, List

from ddgs import DDGS

from app.services.search.base_provider import BaseSearchProvider
from app.services.search.cache import get_cached_results, set_cached_results

logger = logging.getLogger(__name__)

class DDGSearchProvider(BaseSearchProvider):
    
    def __init__(self):
        self.provider_name = "duckduckgo"

    def _sync_search(self, query: str, num_results: int) -> List[dict[str, Any]]:
        results = []
        with DDGS() as ddgs:
            raw_results = ddgs.text(query, region='us-en', safesearch='on', max_results=num_results)
            if raw_results:
                for res in raw_results:
                    results.append({
                        "title": res.get("title", ""),
                        "url": res.get("href", ""),
                        "snippet": res.get("body", "")
                    })
        return results

    async def search(self, query: str, num_results: int = 5) -> List[dict[str, Any]]:
        # 1. Check cache
        cached = get_cached_results(self.provider_name, query)
        if cached is not None:
            logger.info("Search cache hit for query: '%s'", query)
            return cached[:num_results]

        logger.info("Search cache miss for query: '%s'. Executing real search.", query)
        
        # 2. Execute search using duckduckgo_search
        try:
            results = await asyncio.to_thread(self._sync_search, query, num_results)
                
            logger.info("DDG parser found %d results", len(results))
            
            # 3. Cache and return
            if results:
                set_cached_results(self.provider_name, query, results)
            return results
            
        except Exception as e:
            logger.error("DDG Search failed for query '%s': %s", query, e)
            return []
