"""Evidence schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from app.schemas import TruthLensBase


class EvidenceResponse(TruthLensBase):
    id: str
    claim_id: str
    document_id: str
    evidence_text: str
    support_score: float
    contradiction_score: float
    relevance_score: float
    retrieval_method: str
    contradiction_indicators_json: Optional[Dict[str, Any]] = None
    created_at: datetime
