"""Unit tests for Phase 4 Document Extractor."""

import pytest
from app.services.retrieval.document_extractor import fetch_and_extract_article

@pytest.mark.asyncio
async def test_fetch_and_extract_article():
    # Wikipedia is a stable site to test extraction
    url = "https://en.wikipedia.org/wiki/France"
    
    # Run extractor
    result = await fetch_and_extract_article(url)
    
    # Verify result
    assert result is not None
    assert "France" in result["text"] or "French Republic" in result["text"]
    assert "Paris" in result["text"]  # Content should be extracted
    assert result["url"] == url
    assert "retrieved_at" in result
