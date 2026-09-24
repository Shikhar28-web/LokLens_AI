"""Verdict and full report schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.schemas import TruthLensBase
from app.schemas.claim import ClaimResponse
from app.schemas.evidence import EvidenceResponse
from app.schemas.image import ImageResponse


class SourceResponse(TruthLensBase):
    id: str
    url: str
    domain: str
    source_type: str
    authority_score: float
    quality_score: float
    quality_components_json: Optional[Dict[str, Any]] = None


class TimelineEventResponse(TruthLensBase):
    id: str
    event_date: Optional[str] = None
    event_description: str
    source_id: Optional[str] = None
    document_id: Optional[str] = None


class GraphNodeResponse(TruthLensBase):
    node_type: str
    node_id: str
    label: str


class GraphEdgeResponse(TruthLensBase):
    from_node_type: str
    from_node_id: str
    to_node_type: str
    to_node_id: str
    relationship_type: str
    weight: float


class GraphResponse(TruthLensBase):
    nodes: List[GraphNodeResponse]
    edges: List[GraphEdgeResponse]


class VerdictResponse(TruthLensBase):
    id: str
    submission_id: str
    claim_verdict: Optional[str] = None
    claim_confidence: Optional[float] = None
    image_verdict: Optional[str] = None
    image_confidence: Optional[float] = None
    overall_status: Optional[str] = None
    support_score: Optional[float] = None
    contradiction_score: Optional[float] = None
    source_quality_score: Optional[float] = None
    independent_source_count: Optional[int] = None
    duplicate_source_group_count: Optional[int] = None
    multimodal_consistency_score: Optional[float] = None
    explanation_json: Optional[Dict[str, Any]] = None
    score_components_json: Optional[Dict[str, Any]] = None
    limitations_json: Optional[List[str]] = None
    created_at: datetime


class FullReportResponse(TruthLensBase):
    """Complete verification report returned by GET /api/submissions/{id}/report."""
    submission_id: str
    overall_status: Optional[str] = None
    claim_verdict: Optional[str] = None
    claim_confidence: Optional[float] = None
    image_verdict: Optional[str] = None
    image_confidence: Optional[float] = None
    summary: Optional[str] = None

    claims: List[ClaimResponse] = []
    supporting_evidence: List[EvidenceResponse] = []
    contradicting_evidence: List[EvidenceResponse] = []
    sources: List[SourceResponse] = []
    images: List[ImageResponse] = []

    independent_source_count: Optional[int] = None
    duplicate_source_group_count: Optional[int] = None
    multimodal_consistency_score: Optional[float] = None

    timeline: List[TimelineEventResponse] = []
    limitations: List[str] = []
    score_components: Optional[Dict[str, Any]] = None
