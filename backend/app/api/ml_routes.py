"""
ML Inference API Endpoints
===========================
Exposes direct inference for each of the 4 independent models:
1. POST /api/v1/ml/verify-claim            -> FEVER claim verification
2. POST /api/v1/ml/classify-news           -> FakeNewsNet news article classifier
3. POST /api/v1/ml/image-text-consistency  -> NewsCLIPpings image-text consistency
4. POST /api/v1/ml/video-verify            -> FakeTT multimodal video classifier
5. POST /api/v1/ml/unified-route           -> Unified Multimodal Router
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from ml.inference.unified_router import get_router

router = APIRouter()


class ClaimVerificationRequest(BaseModel):
    claim: str = Field(..., description="The factual claim to verify")
    evidence: str = Field(..., description="The evidence passage or sentence to verify against")


class NewsClassificationRequest(BaseModel):
    text: str = Field(..., description="Full text content of the news article")
    title: Optional[str] = Field("", description="Optional article headline")


class ImageTextConsistencyRequest(BaseModel):
    caption: str = Field(..., description="News caption or claim text associated with the image")
    image_path: Optional[str] = Field(None, description="Absolute or relative path to the image file")


class UnifiedRouteRequest(BaseModel):
    text: Optional[str] = None
    title: Optional[str] = None
    claims: Optional[List[str]] = None
    evidence_texts: Optional[List[str]] = None
    image_path: Optional[str] = None


@router.post("/verify-claim", summary="FEVER Claim-Evidence Verification")
def verify_claim(req: ClaimVerificationRequest):
    """
    Classify whether retrieved evidence SUPPORTS, REFUTES, or provides NOT ENOUGH INFO for a claim.
    """
    try:
        from ml.inference.fever_inference import predict
        return predict(req.claim, req.evidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify-news", summary="FakeNewsNet News Article Classifier")
def classify_news(req: NewsClassificationRequest):
    """
    Classify whether a news article is REAL or FAKE using TF-IDF n-grams and NER features.
    """
    try:
        from ml.inference.fakenewsnet_inference import predict
        return predict(req.text, title=req.title or "")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image-text-consistency", summary="NewsCLIPpings Consistency Detection")
def image_text_consistency(req: ImageTextConsistencyRequest):
    """
    Determine whether an image is contextually MATCHED or MISMATCHED with its caption.
    """
    try:
        from ml.inference.newsclippings_inference import predict
        return predict(image_path=req.image_path, caption=req.caption)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unified-route", summary="Unified Multimodal ML Router")
def unified_route(req: UnifiedRouteRequest):
    """
    Single multimodal entry point: routes text, image, and video to their respective models
    and returns a combined, calibrated evaluation report.
    """
    try:
        unified_router = get_router()
        req_dict = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        return unified_router.route(req_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
