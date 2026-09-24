"""
Abstract Search Provider.
Defines the interface that all search implementations (DDG, SerpAPI, Bing) must follow.
"""

from abc import ABC, abstractmethod
from typing import Any, List

class BaseSearchProvider(ABC):
    
    @abstractmethod
    async def search(self, query: str, num_results: int = 5) -> List[dict[str, Any]]:
        """
        Execute a search query and return a list of result dictionaries.
        Each dictionary should contain at least:
        - url: The result URL
        - title: The page title
        - snippet: The text snippet/description
        """
        pass
