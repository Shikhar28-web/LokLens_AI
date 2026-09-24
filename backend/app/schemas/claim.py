"""Claim schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.schemas import TruthLensBase


class ClaimResponse(TruthLensBase):
    id: str
    submission_id: str
    claim_text: str
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    entities_json: Optional[Dict[str, Any]] = None
    keywords_json: Optional[List[str]] = None
    search_queries_json: Optional[List[str]] = None
    created_at: datetime
