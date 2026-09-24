"""Search router — debug endpoint to run a manual search query."""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 10

    @field_validator("query")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v.strip()

    @field_validator("max_results")
    @classmethod
    def clamp_results(cls, v: Optional[int]) -> int:
        v = v or 10
        return max(1, min(v, 20))


class SearchResultItem(BaseModel):
    rank: int
    title: str
    url: str
    snippet: str
    domain: str


class SearchResponse(BaseModel):
    query: str
    provider: str
    results: List[SearchResultItem]
    cached: bool


@router.post("", response_model=SearchResponse)
async def run_search(
    body: SearchRequest,
    db: AsyncSession = Depends(get_db),
) -> SearchResponse:
    """
    Debug endpoint: run a search query through the configured provider.
    Results are cached. This endpoint is for development/testing only.
    Full search is triggered automatically by the analysis pipeline.
    """
    # Search provider wired in Phase 3
    return SearchResponse(
        query=body.query,
        provider="duckduckgo",
        results=[],
        cached=False,
    )
