"""Image and forensics schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from app.schemas import TruthLensBase


class ImageForensicsResponse(TruthLensBase):
    id: str
    image_id: str
    ela_score: Optional[float] = None
    noise_anomaly: Optional[float] = None
    edge_anomaly: Optional[float] = None
    frequency_anomaly: Optional[float] = None
    compression_anomaly: Optional[float] = None
    metadata_anomaly: Optional[float] = None
    copy_move_score: Optional[float] = None
    ai_likelihood_score: Optional[float] = None
    ai_likelihood_label: Optional[str] = None
    exif_json: Optional[Dict[str, Any]] = None
    forensic_features_json: Optional[Dict[str, Any]] = None
    stats_json: Optional[Dict[str, Any]] = None
    analyzed_at: datetime


class ImageResponse(TruthLensBase):
    id: str
    submission_id: str
    mime_type: str
    file_size_bytes: int
    width: Optional[int] = None
    height: Optional[int] = None
    sha256_hash: Optional[str] = None
    phash: Optional[str] = None
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    forensics: Optional[ImageForensicsResponse] = None
    created_at: datetime
