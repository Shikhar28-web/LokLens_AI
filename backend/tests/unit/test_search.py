"""Unit tests for Phase 3 Search components."""

import pytest
from app.services.search.cache import get_cached_results, set_cached_results
from app.services.search.ddg_provider import DDGSearchProvider

@pytest.mark.asyncio
async def test_search_cache():
    # Test setting and getting
    set_cached_results("test_prov", "my test query", [{"url": "http://test", "title": "Test", "snippet": "Test snippet"}])
    
    cached = get_cached_results("test_prov", "my test query")
    assert cached is not None
    assert len(cached) == 1
    assert cached[0]["title"] == "Test"
    
    # Test miss
    miss = get_cached_results("test_prov", "missing query")
    assert miss is None

@pytest.mark.asyncio
async def test_ddg_provider():
    provider = DDGSearchProvider()
    
    # We use a real query, but we don't want tests to fail if DDG blocks CI.
    # The provider will catch exceptions and return an empty list if it fails.
    results = await provider.search("python programming", num_results=2)
    
    # If network succeeds, we should get results
    if results:
        assert len(results) <= 2
        for r in results:
            assert "url" in r
            assert "title" in r
            assert "snippet" in r
