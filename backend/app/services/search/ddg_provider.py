"""
DuckDuckGo Search Provider.
Uses httpx and BeautifulSoup to scrape lite.duckduckgo.com.
Free, no API key required, highly rate-limited so caching is strictly used.
"""

import logging
import urllib.parse
from typing import Any, List

import httpx
from bs4 import BeautifulSoup

from app.services.search.base_provider import BaseSearchProvider
from app.services.search.cache import get_cached_results, set_cached_results

logger = logging.getLogger(__name__)

class DDGSearchProvider(BaseSearchProvider):
    
    def __init__(self):
        self.provider_name = "duckduckgo"
        # Use lite version for stable, javascript-free HTML
        self.base_url = "https://lite.duckduckgo.com/lite/"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    async def search(self, query: str, num_results: int = 5) -> List[dict[str, Any]]:
        # 1. Check cache
        cached = get_cached_results(self.provider_name, query)
        if cached is not None:
            logger.info("Search cache hit for query: '%s'", query)
            return cached[:num_results]

        logger.info("Search cache miss for query: '%s'. Executing real search.", query)
        results = []
        
        # 2. Execute HTTP request
        try:
            async with httpx.AsyncClient(headers=self.headers, timeout=10.0) as client:
                resp = await client.post(
                    self.base_url,
                    data={"q": query},
                )
                resp.raise_for_status()
                
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # The lite DDG layout usually has results in a table
            # Title is in <td class="result-snippet"> or <a class="result-url">
            # Actually, standard lite format:
            # tr -> td class="result-snippet"
            
            # Robust extraction based on known DDG Lite structure
            # Title/URL are in <a class="result-link">
            # Snippet is in <td class="result-snippet">
            
            result_rows = soup.find_all("tr")
            current_result = {}
            
            for row in result_rows:
                if len(results) >= num_results:
                    break
                
                # Check for title and url link
                link_tag = row.find("a", class_="result-link")
                if link_tag:
                    current_result["title"] = link_tag.get_text(strip=True)
                    current_result["url"] = link_tag.get("href")
                    
                # Check for snippet
                snippet_td = row.find("td", class_="result-snippet")
                if snippet_td:
                    current_result["snippet"] = snippet_td.get_text(strip=True)
                    
                    # If we have all parts, save and reset
                    if current_result.get("url") and current_result.get("title"):
                        results.append(current_result)
                    current_result = {}
            
            # 3. Cache and return
            set_cached_results(self.provider_name, query, results)
            return results[:num_results]

        except Exception as e:
            logger.error("DDG Search failed for query '%s': %s", query, e)
            return []
