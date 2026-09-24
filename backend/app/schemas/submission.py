"""Submission request / response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import field_validator

from app.schemas import TruthLensBase


class SubmissionCreate(TruthLensBase):
    """Body for POST /api/submissions (text field).
    Image is provided as multipart; handled separately."""
    text: Optional[str] = None

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("text must not be blank")
        return v


class SubmissionResponse(TruthLensBase):
    id: str
    input_type: str
    status: str
    raw_text: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
