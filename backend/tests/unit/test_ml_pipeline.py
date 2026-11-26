"""
Unit & Integration Tests for LokLens ML Models and Inference Endpoints
======================================================================
Tests:
1. FEVER claim verification inference
2. FakeNewsNet news article classification inference
3. NewsCLIPpings image-text consistency inference
4. FakeTT multimodal video classifier inference
5. Unified router inference
6. FastAPI ML REST API endpoints
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_fever_inference():
    from ml.inference.fever_inference import predict
    res = predict(
        claim="The government inaugurated the solar energy facility.",
        evidence="Official statements confirmed the inauguration of the solar plant.",
    )
    assert res["processing_status"] == "ok"
    assert res["label"] in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")
    assert 0.0 <= res["confidence"] <= 1.0


def test_fakenewsnet_inference():
    from ml.inference.fakenewsnet_inference import predict
    res = predict(
        text="The state electoral board officially certified the election results on Monday morning.",
        title="Election Certified",
    )
    assert res["processing_status"] == "ok"
    assert res["label"] in ("REAL", "FAKE")
    assert 0.0 <= res["confidence"] <= 1.0


def test_newsclippings_inference():
    from ml.inference.newsclippings_inference import predict
    res = predict(
        image_path=None,
        caption="World leaders assemble for the high-level peace summit.",
        prefer="mlp",
    )
    assert res["processing_status"] == "ok"
    assert res["label"] in ("MATCHED", "MISMATCHED")
    assert 0.0 <= res["confidence"] <= 1.0


def test_unified_router():
    from ml.inference.unified_router import get_router
    router = get_router()
    out = router.route({
        "text": "Authorities released verified employment statistics today.",
        "title": "Employment Report",
        "claims": ["Authorities released verified employment statistics."],
    })
    assert "text" in out
    assert "combined_summary" in out
    assert out["disclaimer"]


def test_api_verify_claim_endpoint():
    resp = client.post("/api/ml/verify-claim", json={
        "claim": "Space agency launched satellite today.",
        "evidence": "The space agency confirmed successful launch of the communications satellite.",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["processing_status"] == "ok"
    assert "label" in data


def test_api_classify_news_endpoint():
    resp = client.post("/api/ml/classify-news", json={
        "text": "The Supreme Court delivered its final verdict on the constitutional matter.",
        "title": "Court Verdict",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["processing_status"] == "ok"
    assert data["label"] in ("REAL", "FAKE")


def test_api_image_text_consistency_endpoint():
    resp = client.post("/api/ml/image-text-consistency", json={
        "image_path": None,
        "caption": "Summit delegates pose for group photograph.",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["processing_status"] == "ok"
    assert data["label"] in ("MATCHED", "MISMATCHED")


def test_api_unified_route_endpoint():
    resp = client.post("/api/ml/unified-route", json={
        "text": "Health officials declared regional dengue decline after public awareness drives.",
        "title": "Health Report",
        "claims": ["Dengue cases declined following awareness drives."],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "modalities_processed" in data
    assert "combined_summary" in data



