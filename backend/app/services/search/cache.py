"""
Search result cache using diskcache.
Prevents hammering search engines and speeds up repeated analysis of the same claims.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import diskcache

from app.config import get_settings

settings = get_settings()

CACHE_DIR = settings.upload_path.parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 24-hour expiration for search results
_cache = diskcache.Cache(str(CACHE_DIR), expire=86400)

def _hash_query(provider: str, query: str) -> str:
    """Generate a consistent cache key for a search query."""
    key_str = f"{provider}:{query.strip().lower()}"
    return hashlib.sha256(key_str.encode()).hexdigest()

def get_cached_results(provider: str, query: str) -> Optional[list[dict[str, Any]]]:
    """Retrieve cached search results if available and not expired."""
    key = _hash_query(provider, query)
    cached_val = _cache.get(key)
    if cached_val is not None:
        return json.loads(cached_val)
    return None

def set_cached_results(provider: str, query: str, results: list[dict[str, Any]]) -> None:
    """Store search results in the cache."""
    key = _hash_query(provider, query)
    _cache.set(key, json.dumps(results))
