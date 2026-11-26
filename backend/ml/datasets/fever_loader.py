"""
FEVER Dataset Loader
====================
Loads the FEVER v1.0 claim-verification dataset from HuggingFace.
Supports both the raw FEVER format (with Wikipedia evidence sentence IDs)
and the NLI-flattened variant (pietrolesci/nli_fever) for direct training.

Labels: SUPPORTS (0) | REFUTES (1) | NOT ENOUGH INFO (2)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

LABEL2ID = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# NLI-flat label mapping from pietrolesci/nli_fever
NLI_LABEL2ID = {"entailment": 0, "contradiction": 1, "neutral": 2}


def _load_config() -> Dict[str, Any]:
    cfg_path = Path(__file__).parent.parent / "config" / "datasets.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def load_fever_raw(
    split: str = "train",
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
) -> Any:
    """
    Load raw FEVER dataset from HuggingFace.
    Each record: {id, claim, label, evidence (list of sentence IDs)}

    Args:
        split: 'train' | 'paper_dev' | 'paper_test'
        cache_dir: local HuggingFace cache directory
        trust_remote_code: required for FEVER custom loading script

    Returns:
        HuggingFace Dataset object
    """
    from datasets import load_dataset

    cfg = _load_config()
    fever_cfg = cfg["datasets"]["fever"]
    cache = cache_dir or fever_cfg["local_cache_dir"]
    os.makedirs(cache, exist_ok=True)

    logger.info("Loading FEVER raw (split=%s) from HuggingFace ...", split)
    ds = load_dataset(
        fever_cfg["huggingface_name"],
        fever_cfg["huggingface_config"],
        split=split,
        cache_dir=cache,
        trust_remote_code=trust_remote_code,
    )
    logger.info("FEVER raw %s: %d examples", split, len(ds))
    return ds


def load_fever_nli(
    split: str = "train",
    cache_dir: Optional[str] = None,
) -> Any:
    """
    Load pietrolesci/nli_fever — a flattened NLI version of FEVER.
    Each record: {premise (evidence), hypothesis (claim), label}
    Labels: entailment=SUPPORTS, contradiction=REFUTES, neutral=NOT ENOUGH INFO

    Returns:
        HuggingFace Dataset object
    """
    from datasets import load_dataset

    cfg = _load_config()
    fever_cfg = cfg["datasets"]["fever"]
    cache = cache_dir or fever_cfg["local_cache_dir"]
    os.makedirs(cache, exist_ok=True)

    logger.info("Loading FEVER NLI variant (split=%s) ...", split)
    ds = load_dataset(
        fever_cfg["nli_variant"],
        split=split,
        cache_dir=cache,
    )
    logger.info("FEVER NLI %s: %d examples", split, len(ds))
    return ds


def load_all_splits(
    use_nli: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[Any, Any, Any]:
    """
    Returns (train, val, test) datasets.
    If use_nli=True, returns the flat NLI format (preferred for training).
    If use_nli=False, returns raw FEVER with evidence sentence IDs.
    """
    cfg = _load_config()
    splits_cfg = cfg["datasets"]["fever"]["splits"]

    if use_nli:
        # nli_fever has train, dev, test
        from datasets import load_dataset
        cache = cache_dir or cfg["datasets"]["fever"]["local_cache_dir"]
        all_ds = load_dataset(cfg["datasets"]["fever"]["nli_variant"], cache_dir=cache)
        train = all_ds.get("train")
        val = all_ds.get("dev") or all_ds.get("validation")
        test = all_ds.get("test")
    else:
        train = load_fever_raw(splits_cfg["train"], cache_dir)
        val = load_fever_raw(splits_cfg["val"], cache_dir)
        test = load_fever_raw(splits_cfg["test"], cache_dir)

    return train, val, test


def validate_dataset(ds: Any, name: str = "FEVER") -> Dict[str, Any]:
    """
    Run integrity checks on a loaded FEVER split.
    Returns a summary dict with counts, label distribution, and any issues.
    """
    issues = []
    total = len(ds)

    if total == 0:
        issues.append("Dataset is empty!")
        return {"total": 0, "issues": issues}

    # Check for required columns
    required = {"claim", "label"} if "claim" in ds.column_names else {"hypothesis", "label"}
    missing = required - set(ds.column_names)
    if missing:
        issues.append(f"Missing columns: {missing}")

    # Label distribution
    label_col = ds.column_names
    labels = ds["label"] if "label" in ds.column_names else []
    from collections import Counter
    label_dist = Counter(labels)

    # Check for null/empty claims
    claim_col = "claim" if "claim" in ds.column_names else "hypothesis"
    empty_claims = sum(1 for c in ds[claim_col] if not c or not str(c).strip())
    if empty_claims > 0:
        issues.append(f"{empty_claims} empty claim/hypothesis strings found.")

    summary = {
        "dataset": name,
        "total": total,
        "columns": ds.column_names,
        "label_distribution": dict(label_dist),
        "empty_claims": empty_claims,
        "issues": issues,
    }
    logger.info("[%s] Validation: %s", name, summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== FEVER Dataset Loader Test ===")
    train, val, test = load_all_splits(use_nli=True)
    for name, ds in [("train", train), ("val", val), ("test", test)]:
        summary = validate_dataset(ds, f"FEVER-NLI-{name}")
        print(summary)

