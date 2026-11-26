"""
NewsCLIPpings Preprocessing Pipeline
=====================================
Handles CLIP embedding extraction (if raw images available)
and feature fusion strategies for the classification stage.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)

LABEL2ID = {"MATCHED": 0, "MISMATCHED": 1, "pristine": 0, "falsified": 1}
ID2LABEL = {0: "MATCHED", 1: "MISMATCHED"}


def _load_config() -> Dict[str, Any]:
    cfg_path = Path(__file__).parent.parent / "config" / "datasets.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# CLIP embedding extraction (Option B: raw images)
# ---------------------------------------------------------------------------

_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = None


def _get_clip(model_name: str = "openai/clip-vit-base-patch32"):
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE
    if _CLIP_MODEL is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        _CLIP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading CLIP model '%s' on %s ...", model_name, _CLIP_DEVICE)
        try:
            _CLIP_MODEL = CLIPModel.from_pretrained(model_name, local_files_only=True).to(_CLIP_DEVICE)
            _CLIP_PREPROCESS = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        except Exception:
            _CLIP_MODEL = CLIPModel.from_pretrained(model_name).to(_CLIP_DEVICE)
            _CLIP_PREPROCESS = CLIPProcessor.from_pretrained(model_name)
        _CLIP_MODEL.eval()
        logger.info("CLIP model loaded.")
    return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE


def embed_images_clip(
    image_paths: List[str],
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract CLIP image embeddings for a list of image file paths.

    Returns:
        (N, 512) numpy array of normalized embeddings
    """
    import torch
    from PIL import Image

    model, processor, device = _get_clip(model_name)
    all_embeddings = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start: start + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as e:
                logger.warning("Failed to load image %s: %s", p, e)
                images.append(Image.new("RGB", (224, 224), color=0))

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            elif hasattr(feats, "image_embeds") and feats.image_embeds is not None:
                feats = feats.image_embeds
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embeddings.append(feats.cpu().numpy())

    return np.vstack(all_embeddings)


def embed_texts_clip(
    texts: List[str],
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Extract CLIP text embeddings for a list of caption strings.

    Returns:
        (N, 512) numpy array of normalized embeddings
    """
    import torch

    model, processor, device = _get_clip(model_name)
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start: start + batch_size]
        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,  # CLIP text encoder limit
        ).to(device)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
            if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                feats = feats.pooler_output
            elif hasattr(feats, "text_embeds") and feats.text_embeds is not None:
                feats = feats.text_embeds
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embeddings.append(feats.cpu().numpy())

    return np.vstack(all_embeddings)


# ---------------------------------------------------------------------------
# Feature normalization
# ---------------------------------------------------------------------------

def normalize_embeddings(X: np.ndarray, scaler=None, fit: bool = True) -> Tuple[np.ndarray, Any]:
    """
    L2-normalize + StandardScaler for embeddings.
    Returns (X_scaled, scaler).
    """
    from sklearn.preprocessing import StandardScaler

    # L2 normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / (norms + 1e-9)

    if scaler is None:
        scaler = StandardScaler()

    X = scaler.fit_transform(X) if fit else scaler.transform(X)
    return X, scaler


def add_cosine_similarity_feature(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
) -> np.ndarray:
    """
    Compute per-sample cosine similarity and append as an extra feature column.
    Input embeddings should already be L2-normalized.
    """
    cos = np.sum(img_emb * txt_emb, axis=1, keepdims=True)
    return cos  # shape (N, 1)


def build_feature_matrix(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    fusion: str = "concat+diff+cos",
    scaler=None,
    fit: bool = True,
) -> Tuple[np.ndarray, Any]:
    """
    Build a rich feature matrix from image and text CLIP embeddings.

    Fusion strategies:
      - 'concat':          [img | txt]
      - 'diff':            img - txt
      - 'cosine':          cosine_sim (scalar)
      - 'concat+diff':     [img | txt | img-txt]
      - 'concat+diff+cos': [img | txt | img-txt | cos]  ← recommended

    Returns:
        (X_features, scaler)
    """
    from sklearn.preprocessing import StandardScaler

    # Normalize
    norms_i = np.linalg.norm(img_emb, axis=1, keepdims=True)
    norms_t = np.linalg.norm(txt_emb, axis=1, keepdims=True)
    img_n = img_emb / (norms_i + 1e-9)
    txt_n = txt_emb / (norms_t + 1e-9)

    cos = np.sum(img_n * txt_n, axis=1, keepdims=True)
    diff = img_n - txt_n

    if fusion == "concat":
        X = np.concatenate([img_n, txt_n], axis=1)
    elif fusion == "diff":
        X = diff
    elif fusion == "cosine":
        X = cos
    elif fusion == "concat+diff":
        X = np.concatenate([img_n, txt_n, diff], axis=1)
    elif fusion == "concat+diff+cos":
        X = np.concatenate([img_n, txt_n, diff, cos], axis=1)
    else:
        raise ValueError(f"Unknown fusion: {fusion}")

    if scaler is not None:
        X = scaler.fit_transform(X) if fit else scaler.transform(X)
    elif fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, scaler


# ---------------------------------------------------------------------------
# MLP Classifier Dataset
# ---------------------------------------------------------------------------

class NewsCLIPDataset:
    """PyTorch Dataset for MLP classifier on CLIP embeddings."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        import torch
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        return self.X[idx], self.y[idx]

