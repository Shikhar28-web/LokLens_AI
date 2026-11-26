"""
NewsCLIPpings Inference Module
================================
Returns:
    {
        "label": "MATCHED" | "MISMATCHED",
        "label_id": 0 | 1,
        "confidence": float,
        "cosine_similarity": float,
        "all_scores": {"MATCHED": f, "MISMATCHED": f},
        "model_type": "mlp" | "classical",
        "processing_status": "ok" | "error",
        "note": str
    }
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

MODEL_DIR = Path("models/newsclippings_image_text")
ID2LABEL = {0: "MATCHED", 1: "MISMATCHED"}

_model = None
_classical_model = None
_model_type = None
_model_config = None
_device = None


def load_model(prefer: str = "mlp"):
    global _model, _classical_model, _model_type, _model_config, _device

    if _model is not None or _classical_model is not None:
        return

    if prefer == "mlp":
        try:
            import torch
            from ml.training.train_newsclippings import MLPClassifier
            cfg_path = MODEL_DIR / "mlp_config.json"
            weights_path = MODEL_DIR / "mlp_best.pt"
            if not cfg_path.exists() or not weights_path.exists():
                raise FileNotFoundError(str(weights_path))
            with open(cfg_path) as f:
                _model_config = json.load(f)
            mlp = MLPClassifier(
                _model_config["input_dim"],
                hidden=_model_config["hidden"],
                num_classes=_model_config["num_classes"],
            )
            mlp.load_state_dict(torch.load(weights_path, map_location="cpu"))
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mlp.to(_device).eval()
            _model = mlp
            _model_type = "mlp"
            logger.info("NewsCLIPpings MLP loaded.")
            return
        except Exception as e:
            logger.warning("MLP load failed (%s), trying classical ...", e)

    clf_path = MODEL_DIR / "classical_model.pkl"
    if not clf_path.exists():
        raise RuntimeError(f"No model found in {MODEL_DIR}. Run training first.")
    with open(clf_path, "rb") as f:
        _classical_model = pickle.load(f)
    _model_type = "classical"
    logger.info("NewsCLIPpings classical model loaded.")


def _get_clip_embeddings(image_path: Optional[str], caption: str):
    from ml.preprocessing.newsclippings_preprocessor import embed_images_clip, embed_texts_clip
    img_emb = embed_images_clip([image_path])[0] if image_path else np.zeros(512, dtype=np.float32)
    txt_emb = embed_texts_clip([caption])[0]
    return img_emb, txt_emb


def predict(
    image_path: Optional[str],
    caption: str,
    prefer: str = "mlp",
) -> Dict[str, Any]:
    """
    Predict image-text consistency.

    Args:
        image_path: path to image file (None → zero embedding)
        caption: associated text caption
        prefer: 'mlp' | 'classical'
    """
    if not caption:
        return {"processing_status": "error", "error": "Empty caption."}

    load_model(prefer)

    try:
        img_emb, txt_emb = _get_clip_embeddings(image_path, caption)

        # Cosine similarity (pre-normalized embeddings)
        img_n = img_emb / (np.linalg.norm(img_emb) + 1e-9)
        txt_n = txt_emb / (np.linalg.norm(txt_emb) + 1e-9)
        cos_sim = float(np.dot(img_n, txt_n))

        # Build feature vector
        from ml.preprocessing.newsclippings_preprocessor import build_feature_matrix
        X, _ = build_feature_matrix(
            img_emb[np.newaxis], txt_emb[np.newaxis],
            fusion="concat+diff+cos", fit=False, scaler=None,
        )

        if _model_type == "mlp":
            return _predict_mlp(X, cos_sim)
        else:
            return _predict_classical(X, cos_sim)
    except Exception as e:
        logger.exception("NewsCLIPpings inference error: %s", e)
        return {"processing_status": "error", "error": str(e)}


def _predict_mlp(X: np.ndarray, cos_sim: float) -> Dict[str, Any]:
    import torch
    with torch.no_grad():
        logits = _model(torch.tensor(X, dtype=torch.float32).to(_device))
        probs = torch.softmax(logits, -1).squeeze().cpu().numpy()
    label_id = int(np.argmax(probs))
    return {
        "label": ID2LABEL[label_id],
        "label_id": label_id,
        "confidence": float(probs[label_id]),
        "cosine_similarity": cos_sim,
        "all_scores": {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))},
        "model_type": "mlp",
        "processing_status": "ok",
        "note": "MISMATCHED means the ML model found embedding-level inconsistency. Not definitive proof of manipulation.",
    }


def _predict_classical(X: np.ndarray, cos_sim: float) -> Dict[str, Any]:
    pred = _classical_model.predict(X)[0]
    proba = _classical_model.predict_proba(X)[0] if hasattr(_classical_model, "predict_proba") else None
    conf = float(max(proba)) if proba is not None else 1.0
    all_scores = {ID2LABEL.get(i, str(i)): float(p) for i, p in enumerate(proba)} if proba is not None else {}
    return {
        "label": ID2LABEL.get(int(pred), str(pred)),
        "label_id": int(pred),
        "confidence": conf,
        "cosine_similarity": cos_sim,
        "all_scores": all_scores,
        "model_type": "classical",
        "processing_status": "ok",
        "note": "Classical model on CLIP embeddings.",
    }


if __name__ == "__main__":
    import pprint
    logging.basicConfig(level=logging.INFO)
    print("Testing NewsCLIPpings MLP prediction...")
    res_mlp = predict(image_path=None, caption="World leaders gather at the global climate summit.", prefer="mlp")
    pprint.pprint(res_mlp)
    print("Testing NewsCLIPpings Classical prediction...")
    res_cla = predict(image_path=None, caption="World leaders gather at the global climate summit.", prefer="classical")
    pprint.pprint(res_cla)

