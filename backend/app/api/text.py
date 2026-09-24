"""Text router — standalone text-only quick analysis endpoint."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter()


class TextAnalyzeRequest(BaseModel):
    text: str
    language: Optional[str] = "en"

    @field_validator("text")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        if len(v) > 10_000:
            raise ValueError("text must be under 10,000 characters")
        return v.strip()


class TextAnalyzeResponse(BaseModel):
    message: str
    text_length: int
    language: str
    submission_hint: str


@router.post("/analyze", response_model=TextAnalyzeResponse)
async def analyze_text(
    body: TextAnalyzeRequest,
    db: AsyncSession = Depends(get_db),
) -> TextAnalyzeResponse:
    """
    Quick text-only analysis stub.
    For full analysis, create a submission via POST /api/submissions.
    """
    return TextAnalyzeResponse(
        message="Text received. Submit via /api/submissions for full pipeline analysis.",
        text_length=len(body.text),
        language=body.language or "en",
        submission_hint="POST /api/submissions with text form field",
    )
