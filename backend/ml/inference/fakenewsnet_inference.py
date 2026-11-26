"""
FakeNewsNet Inference Module
==============================
Returns:
    {
        "label": "FAKE" | "REAL",
        "label_id": 0 | 1,
        "confidence": float,
        "all_scores": {"REAL": f, "FAKE": f},
        "model_type": "transformer" | "baseline",
        "processing_status": "ok" | "error",
        "note": str
    }

IMPORTANT: 'FAKE' prediction means the classifier found stylistic/statistical
patterns associated with fake news. It is NOT proof the article is false.
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/fakenewsnet_classifier")
ID2LABEL = {0: "REAL", 1: "FAKE"}

_model = None
_tokenizer = None
_vectorizer = None
_baseline_model = None
_model_type = None
_device = None


def load_model(prefer: str = "transformer", force_reload: bool = False):
    global _model, _tokenizer, _vectorizer, _baseline_model, _model_type, _device

    if not force_reload:
        if prefer == "transformer" and _model is not None:
            _model_type = "transformer"
            return
        if prefer == "baseline" and _baseline_model is not None:
            _model_type = "baseline"
            return
        if _model is not None and prefer != "baseline":
            _model_type = "transformer"
            return
        if _baseline_model is not None and prefer != "transformer":
            _model_type = "baseline"
            return

    model_path = MODEL_DIR / "best_model"
    if prefer == "transformer" and model_path.exists():
        try:
            import torch
            from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
            _tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_path), local_files_only=True)
            _model = DistilBertForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _model.to(_device).eval()
            _model_type = "transformer"
            logger.info("FakeNewsNet transformer loaded on %s", _device)
            return
        except Exception as e:
            logger.warning("Transformer load failed (%s), falling back to baseline ...", e)

    # Baseline fallback or explicit preference
    vec_path = MODEL_DIR / "baseline_vectorizer.pkl"
    clf_path = MODEL_DIR / "baseline_model.pkl"
    if not vec_path.exists() or not clf_path.exists():
        # If transformer exists, try it before raising error
        if model_path.exists():
            try:
                import torch
                from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
                _tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_path))
                _model = DistilBertForSequenceClassification.from_pretrained(str(model_path))
                _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                _model.to(_device).eval()
                _model_type = "transformer"
                logger.info("FakeNewsNet transformer loaded on %s", _device)
                return
            except Exception:
                pass
        raise RuntimeError(f"No FakeNewsNet model found in {MODEL_DIR}. Run training first.")

    with open(vec_path, "rb") as f:
        _vectorizer = pickle.load(f)
    with open(clf_path, "rb") as f:
        _baseline_model = pickle.load(f)
    _model_type = "baseline"
    logger.info("FakeNewsNet baseline loaded.")


def predict(text: str, title: str = "", prefer: str = "transformer") -> Dict[str, Any]:
    """
    Classify a news article as FAKE or REAL.

    Args:
        text: article body text
        title: article headline (optional, prepended to text)
        prefer: 'transformer' | 'baseline'
    """
    if not text and not title:
        return {"processing_status": "error", "error": "Empty input."}

    load_model(prefer)
    combined = f"{title} . {text}" if title and text else title or text

    try:
        if _model_type == "transformer":
            return _predict_transformer(combined)
        else:
            return _predict_baseline(combined)
    except Exception as e:
        logger.exception("FakeNewsNet inference error: %s", e)
        return {"processing_status": "error", "error": str(e)}


def _predict_transformer(text: str) -> Dict[str, Any]:
    import torch
    from ml.preprocessing.fakenewsnet_preprocessor import clean_article_text

    cleaned = clean_article_text(text, keep_case=True)[:2000]
    enc = _tokenizer(
        cleaned, max_length=512, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = _model(
            input_ids=enc["input_ids"].to(_device),
            attention_mask=enc["attention_mask"].to(_device),
        ).logits
    probs = torch.softmax(logits, -1).squeeze().cpu().numpy()
    label_id = int(np.argmax(probs))
    return {
        "label": ID2LABEL[label_id],
        "label_id": label_id,
        "confidence": float(probs[label_id]),
        "all_scores": {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))},
        "model_type": "transformer",
        "processing_status": "ok",
        "note": (
            "FAKE/REAL prediction reflects stylistic patterns in the text. "
            "This is NOT a factual verification of the article's claims."
        ),
    }


def _predict_baseline(text: str) -> Dict[str, Any]:
    from ml.preprocessing.fakenewsnet_preprocessor import (
        clean_article_text, build_ner_feature_matrix, combine_tfidf_ner
    )
    cleaned = clean_article_text(text, keep_case=False)
    X_tfidf = _vectorizer.transform([cleaned])
    
    # Check if baseline was trained with NER features (37912 vs 37908)
    n_expected = getattr(_baseline_model, "n_features_in_", X_tfidf.shape[1])
    if n_expected > X_tfidf.shape[1]:
        X_ner = build_ner_feature_matrix([text])
        X = combine_tfidf_ner(X_tfidf, X_ner)
    else:
        X = X_tfidf

    pred = int(_baseline_model.predict(X)[0])
    if hasattr(_baseline_model, "predict_proba"):
        proba = _baseline_model.predict_proba(X)[0]
        classes = getattr(_baseline_model, "classes_", [0, 1])
        all_scores = {ID2LABEL.get(int(c), str(c)): float(p) for c, p in zip(classes, proba)}
        conf = float(max(proba))
    else:
        conf = 1.0
        all_scores = {ID2LABEL.get(pred, str(pred)): 1.0}

    return {
        "label": ID2LABEL.get(pred, str(pred)),
        "label_id": pred,
        "confidence": conf,
        "all_scores": all_scores,
        "model_type": "baseline",
        "processing_status": "ok",
        "note": "Baseline TF-IDF + NER model.",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = predict("Breaking: Scientists discover new planet in solar system.", title="Huge Discovery")
    print(json.dumps(result, indent=2))

