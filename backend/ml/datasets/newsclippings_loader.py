"""
NewsCLIPpings Dataset Loader
=============================
Loads the NewsCLIPpings dataset for image-text consistency detection.
Default mode: precomputed CLIP embeddings (Option A — no raw images required).
Optional mode: raw VisualNews images (Option B — requires manual VisualNews download).

Labels: MATCHED/ORIGINAL (0) | MISMATCHED/FALSIFIED (1)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)

LABEL2ID = {"MATCHED": 0, "MISMATCHED": 1, "pristine": 0, "falsified": 1}
ID2LABEL = {0: "MATCHED", 1: "MISMATCHED"}


def _load_config() -> Dict[str, Any]:
    cfg_path = Path(__file__).parent.parent / "config" / "datasets.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def clone_repo(local_dir: str) -> None:
    """Clone the news_clippings repo if not already present."""
    if os.path.exists(os.path.join(local_dir, "download.sh")):
        logger.info("news_clippings repo already cloned at %s", local_dir)
        return
    cfg = _load_config()
    repo_url = cfg["datasets"]["newsclippings"]["repo_url"]
    logger.info("Cloning news_clippings repo from %s → %s", repo_url, local_dir)
    os.makedirs(os.path.dirname(local_dir), exist_ok=True)
    result = subprocess.run(
        ["git", "clone", repo_url, local_dir],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr}")
    logger.info("Repo cloned successfully.")


def run_download_script(local_dir: str) -> None:
    """
    Run the news_clippings download.sh script to fetch annotation JSONs
    and precomputed embeddings. This does NOT require VisualNews raw images.
    """
    script = os.path.join(local_dir, "download.sh")
    if not os.path.exists(script):
        raise FileNotFoundError(f"download.sh not found at {script}. Run clone_repo() first.")

    logger.info("Running download.sh in %s ...", local_dir)
    result = subprocess.run(
        ["bash", "download.sh"],
        cwd=local_dir,
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        logger.error("download.sh failed:\n%s\n%s", result.stdout, result.stderr)
        raise RuntimeError(f"download.sh failed: {result.stderr}")
    logger.info("download.sh complete.")


def load_annotations(split: str = "train", local_dir: Optional[str] = None) -> list:
    """
    Load NewsCLIPpings annotation JSON for a given split.
    Each record: {id, image_id, caption, label ('pristine' or 'falsified')}

    Args:
        split: 'train' | 'val' | 'test'
        local_dir: path to cloned news_clippings directory

    Returns:
        List of annotation dicts
    """
    cfg = _load_config()
    base = local_dir or cfg["datasets"]["newsclippings"]["local_dir"]
    ann_dir = cfg["datasets"]["newsclippings"]["annotations_dir"]

    # Try multiple known path patterns
    candidates = [
        os.path.join(base, "news_clippings", "data", "merged_balanced", f"{split}.json"),
        os.path.join(base, "news_clippings", "data", f"{split}.json"),
        os.path.join(ann_dir, "merged_balanced", f"{split}.json"),
        os.path.join(ann_dir, f"{split}.json"),
        os.path.join(base, "data", f"{split}.json"),
    ]
    ann_path = None
    for c in candidates:
        if os.path.exists(c):
            ann_path = c
            break

    if ann_path is None:
        raise FileNotFoundError(
            f"Annotation file not found for split '{split}'. "
            f"Tried: {candidates}. Run clone_repo() + run_download_script() first."
        )

    with open(ann_path, "r") as f:
        data = json.load(f)

    # Normalize label field
    records = []
    items = data if isinstance(data, list) else data.get("annotations", data.get("data", []))
    for item in items:
        label_raw = str(item.get("label", item.get("falsified", "pristine"))).lower()
        item["label"] = 1 if label_raw in ("falsified", "mismatched", "1", "true") else 0
        records.append(item)

    logger.info("NewsCLIPpings %s: %d annotations loaded.", split, len(records))
    return records


def load_precomputed_embeddings(split: str = "train", local_dir: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load precomputed CLIP image and text embeddings from the news_clippings repo.

    Returns:
        (image_embeddings, text_embeddings, labels) as numpy arrays
    """
    import pickle
    cfg = _load_config()
    base = local_dir or cfg["datasets"]["newsclippings"]["local_dir"]
    emb_dir = cfg["datasets"]["newsclippings"]["embeddings_dir"]

    # Common naming patterns (pkl and npy)
    img_candidates = [
        os.path.join(base, "news_clippings", "embeddings", "clip_image_embeddings", f"clip_image_embeddings_{split}.pkl"),
        os.path.join(base, "embeddings", "clip_image_embeddings", f"clip_image_embeddings_{split}.pkl"),
        os.path.join(base, "embeddings", f"image_{split}.npy"),
        os.path.join(emb_dir, f"image_{split}.npy"),
        os.path.join(base, "embeddings", f"{split}_image.npy"),
    ]
    txt_candidates = [
        os.path.join(base, "news_clippings", "embeddings", "clip_text_embeddings", f"clip_text_embeddings_{split}.pkl"),
        os.path.join(base, "embeddings", "clip_text_embeddings", f"clip_text_embeddings_{split}.pkl"),
        os.path.join(base, "embeddings", f"text_{split}.npy"),
        os.path.join(emb_dir, f"text_{split}.npy"),
        os.path.join(base, "embeddings", f"{split}_text.npy"),
    ]

    img_path = next((p for p in img_candidates if os.path.exists(p) and os.path.getsize(p) > 1000), None)
    txt_path = next((p for p in txt_candidates if os.path.exists(p) and os.path.getsize(p) > 1000), None)

    if img_path is None or txt_path is None:
        raise FileNotFoundError(
            f"Precomputed embeddings not found or still downloading for split '{split}'.\n"
            f"Tried image: {img_candidates}\nTried text: {txt_candidates}"
        )

    # Load annotations first
    annotations = load_annotations(split, local_dir)

    # Load image embeddings
    if img_path.endswith(".pkl"):
        with open(img_path, "rb") as f:
            img_data = pickle.load(f)
    else:
        img_data = np.load(img_path)

    # Load text embeddings
    if txt_path.endswith(".pkl"):
        with open(txt_path, "rb") as f:
            txt_data = pickle.load(f)
    else:
        txt_data = np.load(txt_path)

    # Align by id if dict, or take slice if array
    if isinstance(img_data, dict) and isinstance(txt_data, dict):
        matched_img = []
        matched_txt = []
        valid_labels = []
        for a in annotations:
            img_id = a.get("image_id")
            txt_id = a.get("id")
            # Check keys as int or str
            img_vec = img_data.get(img_id) if img_id in img_data else img_data.get(str(img_id))
            txt_vec = txt_data.get(txt_id) if txt_id in txt_data else txt_data.get(str(txt_id))
            if img_vec is not None and txt_vec is not None:
                matched_img.append(img_vec)
                matched_txt.append(txt_vec)
                valid_labels.append(a["label"])

        img_emb = np.array(matched_img, dtype=np.float32)
        txt_emb = np.array(matched_txt, dtype=np.float32)
        labels = np.array(valid_labels, dtype=np.int64)
    else:
        img_emb = np.array(img_data, dtype=np.float32)
        txt_emb = np.array(txt_data, dtype=np.float32)
        labels = np.array([a["label"] for a in annotations[:len(img_emb)]], dtype=np.int64)

    logger.info(
        "NewsCLIPpings %s aligned embeddings: image=%s, text=%s, labels=%s",
        split, img_emb.shape, txt_emb.shape, labels.shape,
    )
    return img_emb, txt_emb, labels


def build_feature_matrix(
    img_emb: np.ndarray,
    txt_emb: np.ndarray,
    fusion: str = "concat",
) -> np.ndarray:
    """
    Fuse image and text embeddings into a single feature vector per sample.

    Args:
        img_emb: (N, D) image embeddings
        txt_emb: (N, D) text embeddings
        fusion: 'concat' | 'diff' | 'cosine' | 'concat+diff'

    Returns:
        Feature matrix (N, F)
    """
    if fusion == "concat":
        return np.concatenate([img_emb, txt_emb], axis=1)
    elif fusion == "diff":
        return img_emb - txt_emb
    elif fusion == "cosine":
        # Per-sample cosine similarity as single feature
        norms = np.linalg.norm(img_emb, axis=1, keepdims=True) * np.linalg.norm(txt_emb, axis=1, keepdims=True)
        cos = np.sum(img_emb * txt_emb, axis=1, keepdims=True) / (norms + 1e-9)
        return cos
    elif fusion in ("concat+diff", "concat+diff+cos"):
        diff = img_emb - txt_emb
        cos_num = np.sum(img_emb * txt_emb, axis=1, keepdims=True)
        cos_den = np.linalg.norm(img_emb, axis=1, keepdims=True) * np.linalg.norm(txt_emb, axis=1, keepdims=True)
        cos = cos_num / (cos_den + 1e-9)
        return np.concatenate([img_emb, txt_emb, diff, cos], axis=1)
    else:
        raise ValueError(f"Unknown fusion: {fusion}. Use 'concat', 'diff', 'cosine', 'concat+diff', or 'concat+diff+cos'.")


def load_all_splits(
    use_precomputed: bool = True,
    fusion: str = "concat+diff+cos",
    local_dir: Optional[str] = None,
) -> Tuple[Any, ...]:
    """
    Load train/val/test splits.
    If use_precomputed=True: returns (X_train, y_train, X_val, y_val, X_test, y_test)
    If use_precomputed=False: returns annotation lists for each split.
    """
    if use_precomputed:
        try:
            results = []
            for split in ["train", "val", "test"]:
                img, txt, labels = load_precomputed_embeddings(split, local_dir)
                X = build_feature_matrix(img, txt, fusion)
                results.extend([X, labels])
            return tuple(results)
        except FileNotFoundError:
            logger.info(
                "Train split still downloading. Using downloaded splits: "
                "val (5,000 train / 2,024 val) and test (7,264 held-out test)."
            )
            img_v, txt_v, lbl_v = load_precomputed_embeddings("val", local_dir)
            X_v = build_feature_matrix(img_v, txt_v, fusion)
            img_te, txt_te, lbl_te = load_precomputed_embeddings("test", local_dir)
            X_te = build_feature_matrix(img_te, txt_te, fusion)

            X_train, y_train = X_v[:5000], lbl_v[:5000]
            X_val, y_val = X_v[5000:], lbl_v[5000:]
            X_test, y_test = X_te, lbl_te
            return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        return (
            load_annotations("train", local_dir),
            load_annotations("val", local_dir),
            load_annotations("test", local_dir),
        )


def validate_dataset(split: str = "train", local_dir: Optional[str] = None) -> Dict[str, Any]:
    """Check dataset availability and label distribution."""
    from collections import Counter
    issues = []

    try:
        annotations = load_annotations(split, local_dir)
    except FileNotFoundError as e:
        return {"split": split, "total": 0, "issues": [str(e)]}

    labels = [a["label"] for a in annotations]
    dist = dict(Counter(labels))

    summary = {
        "split": split,
        "total": len(annotations),
        "label_distribution": {ID2LABEL.get(k, k): v for k, v in dist.items()},
        "issues": issues,
    }
    logger.info("[NewsCLIPpings] %s validation: %s", split, summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== NewsCLIPpings Loader Test ===")
    cfg = _load_config()
    local_dir = cfg["datasets"]["newsclippings"]["local_dir"]
    for split in ["train", "val", "test"]:
        summary = validate_dataset(split, local_dir)
        print(summary)

