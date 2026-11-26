"""
FEVER Preprocessing Pipeline
==============================
Prepares FEVER NLI data for training.
- Premise (evidence) + Hypothesis (claim) pair tokenization
- Label normalization: entailment→0, contradiction→1, neutral→2
- Deduplication
- TF-IDF feature extraction for baseline ML models
- Transformer tokenization for fine-tuned NLI model
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# NLI-flat label normalization
NLI_LABEL2ID = {
    "entailment": 0,
    "contradiction": 1,
    "neutral": 2,
    "supports": 0,
    "refutes": 1,
    "not enough info": 2,
}
ID2LABEL = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}


def _load_config() -> Dict[str, Any]:
    cfg_path = Path(__file__).parent.parent / "config" / "datasets.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Light cleaning preserving entity casing (needed for NLI)."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def build_input_pair(claim: str, evidence: str) -> str:
    """Format claim+evidence as a single string for TF-IDF baseline."""
    return f"{clean_text(claim)} [SEP] {clean_text(evidence)}"


# ---------------------------------------------------------------------------
# Baseline: TF-IDF features
# ---------------------------------------------------------------------------

def build_tfidf_features(
    pairs: List[Tuple[str, str]],
    labels: List[int],
    vectorizer=None,
    fit: bool = True,
    max_features: int = 50_000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[Any, List[int], Any]:
    """
    Convert (claim, evidence) pairs to TF-IDF features.

    Args:
        pairs: list of (claim, evidence) tuples
        labels: list of integer labels
        vectorizer: fitted TfidfVectorizer (None → create new)
        fit: if True, fit the vectorizer on pairs
        max_features: vocabulary size cap
        ngram_range: n-gram range for TF-IDF

    Returns:
        (X, y, vectorizer)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [build_input_pair(c, e) for c, e in pairs]

    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            min_df=2,
        )

    X = vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts)
    return X, labels, vectorizer


# ---------------------------------------------------------------------------
# Transformer tokenization
# ---------------------------------------------------------------------------

class FEVERDataset:
    """
    PyTorch Dataset for FEVER NLI training.
    Tokenizes (claim, evidence) pairs with a HuggingFace tokenizer.
    """

    def __init__(
        self,
        claims: List[str],
        evidences: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int = 512,
    ):
        self.claims = claims
        self.evidences = evidences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import torch

        claim = clean_text(self.claims[idx])
        evidence = clean_text(self.evidences[idx])

        encoding = self.tokenizer(
            claim,
            evidence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros(self.max_length, dtype=torch.long)).squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def normalize_label(label: Any) -> int:
    """Normalize various label representations to 0/1/2."""
    if isinstance(label, int) and label in (0, 1, 2):
        return label
    return NLI_LABEL2ID.get(str(label).lower().strip(), 2)


def prepare_datasets(
    train_ds: Any,
    val_ds: Any,
    test_ds: Any,
    tokenizer: Optional[Any] = None,
    max_length: int = 512,
) -> Tuple[FEVERDataset, FEVERDataset, FEVERDataset]:
    """
    Convert HuggingFace FEVER NLI splits into FEVERDataset objects.

    Handles both formats:
    - nli_fever:  {'premise': evidence, 'hypothesis': claim, 'label': int}
    - raw FEVER:  {'claim': ..., 'label': str, 'evidence': ...}
    """
    def _extract(ds) -> Tuple[List[str], List[str], List[int]]:
        claims, evidences, labels = [], [], []

        # Determine column layout
        cols = ds.column_names
        if "hypothesis" in cols and "premise" in cols:
            # In pietrolesci/nli_fever: premise=claim, hypothesis=evidence,
            # and integer label is 0=entailment(SUPPORTS), 1=neutral(NEI), 2=contradiction(REFUTES).
            has_gold = "fever_gold_label" in cols
            nli_int_remap = {0: 0, 1: 2, 2: 1}
            for row in ds:
                claims.append(row["premise"])
                evidences.append(row["hypothesis"])
                if has_gold and row.get("fever_gold_label"):
                    labels.append(normalize_label(row["fever_gold_label"]))
                else:
                    raw_lbl = row["label"]
                    labels.append(nli_int_remap.get(raw_lbl, normalize_label(raw_lbl)))
        elif "claim" in cols:
            # Raw FEVER — evidence field is list of sentence annotations
            for row in ds:
                claim = row["claim"]
                # Flatten evidence sentences (may be nested lists)
                ev_parts = []
                raw_ev = row.get("evidence", [])
                if isinstance(raw_ev, list):
                    for ev_group in raw_ev:
                        if isinstance(ev_group, list):
                            for ev_item in ev_group:
                                if isinstance(ev_item, (list, tuple)) and len(ev_item) >= 3:
                                    # [annotation_id, evidence_id, page, sent_id]
                                    # We use page title as fallback text
                                    ev_parts.append(str(ev_item[2]) if len(ev_item) > 2 else "")
                                elif isinstance(ev_item, str):
                                    ev_parts.append(ev_item)
                evidence_text = " ".join(ev_parts).strip() or "[NO EVIDENCE]"
                claims.append(claim)
                evidences.append(evidence_text)
                labels.append(normalize_label(row["label"]))
        else:
            raise ValueError(f"Unknown FEVER dataset column layout: {cols}")

        return claims, evidences, labels

    train_claims, train_ev, train_labels = _extract(train_ds)
    val_claims, val_ev, val_labels = _extract(val_ds)
    test_claims, test_ev, test_labels = _extract(test_ds)

    # Deduplication on train set
    seen = set()
    dedup_idx = []
    for i, (c, e) in enumerate(zip(train_claims, train_ev)):
        key = (c.strip().lower(), e.strip().lower()[:200])
        if key not in seen:
            seen.add(key)
            dedup_idx.append(i)
    orig_len = len(train_claims)
    train_claims = [train_claims[i] for i in dedup_idx]
    train_ev = [train_ev[i] for i in dedup_idx]
    train_labels = [train_labels[i] for i in dedup_idx]
    logger.info("FEVER dedup: %d → %d train examples", orig_len, len(train_claims))

    if tokenizer is None:
        return (
            (train_claims, train_ev, train_labels),
            (val_claims, val_ev, val_labels),
            (test_claims, test_ev, test_labels),
        )

    train_dataset = FEVERDataset(train_claims, train_ev, train_labels, tokenizer, max_length)
    val_dataset = FEVERDataset(val_claims, val_ev, val_labels, tokenizer, max_length)
    test_dataset = FEVERDataset(test_claims, test_ev, test_labels, tokenizer, max_length)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from ml.datasets.fever_loader import load_all_splits
    train, val, test = load_all_splits(use_nli=True)
    result = prepare_datasets(train, val, test)
    print("FEVER preprocessing OK — train:", len(result[0][2]), "examples")

