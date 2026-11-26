"""
FEVER Inference Module
========================
Loads the trained FEVER claim verifier and provides a structured inference function.

Returns:
    {
        "label": "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO",
        "label_id": 0 | 1 | 2,
        "confidence": float,          # softmax probability of predicted class
        "all_scores": {"SUPPORTS": f, "REFUTES": f, "NOT ENOUGH INFO": f},
        "model_type": "transformer" | "baseline",
        "processing_status": "ok" | "error",
        "note": str                   # calibration disclaimer
    }

IMPORTANT: Confidence scores reflect the model's output distribution,
NOT the probability that a claim is factually true in the real world.
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/fever_claim_verifier")
ID2LABEL = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

_model = None
_tokenizer = None
_vectorizer = None
_baseline_model = None
_model_type = None
_device = None


def _load_transformer_model():
    global _model, _tokenizer, _device, _model_type
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = MODEL_DIR / "best_model"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained FEVER transformer not found at {model_path}. "
            "Run: python ml/training/train_fever.py --mode transformer"
        )
    _tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    _model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(_device).eval()
    _model_type = "transformer"
    logger.info("FEVER transformer loaded from %s on %s", model_path, _device)


def _load_baseline_model():
    global _vectorizer, _baseline_model, _model_type

    vec_path = MODEL_DIR / "baseline_vectorizer.pkl"
    model_path = MODEL_DIR / "baseline_model.pkl"

    if not vec_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Trained FEVER baseline not found at {MODEL_DIR}. "
            "Run: python ml/training/train_fever.py --mode baseline"
        )
    with open(vec_path, "rb") as f:
        _vectorizer = pickle.load(f)
    with open(model_path, "rb") as f:
        _baseline_model = pickle.load(f)
    _model_type = "baseline"
    logger.info("FEVER baseline model loaded from %s", MODEL_DIR)


def load_model(prefer: str = "transformer"):
    """
    Load the FEVER model. Tries transformer if present, otherwise uses baseline.
    """
    global _model_type
    if _model is not None or _baseline_model is not None:
        return  # already loaded

    if prefer == "transformer" and (MODEL_DIR / "best_model").exists():
        try:
            _load_transformer_model()
            return
        except Exception as e:
            logger.warning("Transformer load failed (%s), falling back to baseline.", e)
    try:
        _load_baseline_model()
    except Exception as e:
        raise RuntimeError(f"No FEVER model available: {e}")


def predict(
    claim: str,
    evidence: str,
    prefer: str = "transformer",
) -> Dict[str, Any]:
    """
    Run FEVER claim verification inference.

    Args:
        claim: the claim text to verify
        evidence: the evidence text to compare against
        prefer: 'transformer' | 'baseline'

    Returns:
        Structured prediction dict
    """
    if not claim or not claim.strip():
        return {"processing_status": "error", "error": "Empty claim provided."}
    if not evidence or not evidence.strip():
        evidence = "[NO EVIDENCE]"

    load_model(prefer)

    try:
        if _model_type == "transformer":
            return _predict_transformer(claim, evidence)
        else:
            return _predict_baseline(claim, evidence)
    except Exception as e:
        logger.exception("FEVER inference failed: %s", e)
        return {"processing_status": "error", "error": str(e)}


def _predict_transformer(claim: str, evidence: str) -> Dict[str, Any]:
    import torch

    encoding = _tokenizer(
        claim, evidence,
        max_length=512, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = _model(
            input_ids=encoding["input_ids"].to(_device),
            attention_mask=encoding["attention_mask"].to(_device),
        ).logits

    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    label_id = int(np.argmax(probs))
    confidence = float(probs[label_id])

    return {
        "label": ID2LABEL[label_id],
        "label_id": label_id,
        "confidence": confidence,
        "all_scores": {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))},
        "model_type": "transformer",
        "processing_status": "ok",
        "note": (
            "Confidence reflects the model's output distribution on this claim-evidence pair. "
            "It is NOT the probability that the claim is factually true."
        ),
    }


def _predict_baseline(claim: str, evidence: str) -> Dict[str, Any]:
    from ml.preprocessing.fever_preprocessor import build_input_pair

    text = build_input_pair(claim, evidence)
    X = _vectorizer.transform([text])
    pred = _baseline_model.predict(X)[0]
    proba = _baseline_model.predict_proba(X)[0] if hasattr(_baseline_model, "predict_proba") else None

    all_scores = {}
    if proba is not None:
        for i, p in enumerate(proba):
            all_scores[ID2LABEL.get(i, str(i))] = float(p)
        confidence = float(max(proba))
    else:
        confidence = 1.0

    return {
        "label": ID2LABEL.get(pred, str(pred)),
        "label_id": int(pred),
        "confidence": confidence,
        "all_scores": all_scores,
        "model_type": "baseline",
        "processing_status": "ok",
        "note": "Baseline model — confidence scores are less calibrated than transformer model.",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = predict(
        claim="The Eiffel Tower is located in Paris, France.",
        evidence="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
    )
    print(json.dumps(result, indent=2))

