"""
FakeNewsNet Preprocessing Pipeline
=====================================
Text cleaning, tokenization, TF-IDF features, lemmatization,
stopword removal, NER feature extraction, and Transformer tokenization.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)

LABEL2ID = {"REAL": 0, "FAKE": 1}
ID2LABEL = {0: "REAL", 1: "FAKE"}


def _load_config() -> Dict[str, Any]:
    cfg_path = Path(__file__).parent.parent / "config" / "datasets.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_CONTRACTION_MAP = {
    r"\bdon't\b": "do not", r"\bdoesn't\b": "does not", r"\bdidn't\b": "did not",
    r"\bcan't\b": "cannot", r"\bwon't\b": "will not", r"\bisn't\b": "is not",
    r"\baren't\b": "are not", r"\bwasn't\b": "was not", r"\bweren't\b": "were not",
    r"\bhaven't\b": "have not", r"\bhasn't\b": "has not", r"\bhadn't\b": "had not",
    r"\bit's\b": "it is", r"\bthat's\b": "that is",
}


def clean_article_text(text: str, keep_case: bool = False) -> str:
    """
    Clean news article text for NLP processing.
    - Remove HTML, URLs, emails
    - Expand contractions
    - Normalize whitespace
    - Optionally lowercase (keep_case=True preserves case for NER)
    """
    if not text or not isinstance(text, str):
        return ""
    # HTML
    text = re.sub(r"<[^>]+>", " ", text)
    # URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Email
    text = re.sub(r"\S+@\S+", " ", text)
    # Strip wire-service datelines (e.g., "WASHINGTON (Reuters) - ", "BRUSSELS — ")
    # at start of text or immediately after the "title . " separator
    text = re.sub(
        r"(^|\.\s+)[A-Z][A-Z\s,./-]{1,35}\s*(?:\([^)]{1,30}\))?\s*[-–—:]\s+",
        r"\1",
        text,
    )
    # Strip explicit wire agency tags like (Reuters), (AP), (AFP)
    text = re.sub(r"\((?:Reuters|AP|AFP|PTI)\)", " ", text, flags=re.IGNORECASE)
    # Contractions
    for pat, repl in _CONTRACTION_MAP.items():
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)
    # Non-printable
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)
    # Excessive punctuation/special chars
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"-]", " ", text)
    # Whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if not keep_case:
        text = text.lower()
    return text


def combine_title_text(row: Dict[str, Any]) -> str:
    """Combine title and text columns with a separator."""
    title = str(row.get("title", "") or "")
    text = str(row.get("text", "") or "")
    combined = f"{title} . {text}" if title and text else title or text
    return combined


# ---------------------------------------------------------------------------
# NLP Pipeline (spaCy-based)
# ---------------------------------------------------------------------------

_NLP_MODEL = None


def _get_nlp():
    global _NLP_MODEL
    if _NLP_MODEL is None:
        import spacy
        try:
            _NLP_MODEL = spacy.load("en_core_web_sm", disable=["parser"])
        except OSError:
            logger.warning("en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")
            _NLP_MODEL = None
    return _NLP_MODEL


def lemmatize_and_remove_stopwords(text: str) -> str:
    """Apply lemmatization and stopword removal using spaCy."""
    nlp = _get_nlp()
    if nlp is None:
        return text
    doc = nlp(text[:5000])  # limit for speed
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and len(token.text) > 2
    ]
    return " ".join(tokens)


def extract_ner_features(text: str) -> Dict[str, Any]:
    """Extract named entity features from text for ML models."""
    nlp = _get_nlp()
    if nlp is None:
        return {"ner_count": 0, "person_count": 0, "org_count": 0, "gpe_count": 0}
    doc = nlp(text[:5000])
    ents = [(e.label_, e.text) for e in doc.ents]
    return {
        "ner_count": len(ents),
        "person_count": sum(1 for label, _ in ents if label == "PERSON"),
        "org_count": sum(1 for label, _ in ents if label == "ORG"),
        "gpe_count": sum(1 for label, _ in ents if label == "GPE"),
        "entities": [t for _, t in ents[:20]],
    }


# ---------------------------------------------------------------------------
# TF-IDF Features (for baseline ML models)
# ---------------------------------------------------------------------------

def build_tfidf_features(
    texts: List[str],
    labels: List[int],
    vectorizer=None,
    fit: bool = True,
    max_features: int = 100_000,
    ngram_range: Tuple[int, int] = (1, 3),
    use_lemma: bool = False,
) -> Tuple[Any, List[int], Any]:
    """
    Build TF-IDF feature matrix.

    Args:
        texts: list of raw article texts
        labels: list of integer labels
        vectorizer: fitted TfidfVectorizer or None
        fit: fit the vectorizer on texts
        max_features: vocabulary cap
        ngram_range: n-gram range
        use_lemma: apply lemmatization before vectorization

    Returns:
        (X_sparse, labels, vectorizer)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    processed = []
    for t in texts:
        t = clean_article_text(t, keep_case=False)
        if use_lemma:
            t = lemmatize_and_remove_stopwords(t)
        processed.append(t)

    if vectorizer is None:
        kwargs = dict(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        if len(processed) > 5:
            kwargs.update(dict(min_df=2, max_df=0.95))
        vectorizer = TfidfVectorizer(**kwargs)

    X = vectorizer.fit_transform(processed) if fit else vectorizer.transform(processed)
    return X, labels, vectorizer


def build_ner_feature_matrix(texts: List[str]) -> np.ndarray:
    """Extract NER-based numerical features (counts) as a dense matrix."""
    rows = []
    for text in texts:
        feats = extract_ner_features(clean_article_text(text, keep_case=True))
        rows.append([
            feats["ner_count"],
            feats["person_count"],
            feats["org_count"],
            feats["gpe_count"],
        ])
    return np.array(rows, dtype=np.float32)


def combine_tfidf_ner(X_tfidf: Any, X_ner: np.ndarray) -> Any:
    """Horizontally stack TF-IDF sparse matrix with dense NER features."""
    from scipy.sparse import hstack, csr_matrix
    return hstack([X_tfidf, csr_matrix(X_ner)])


# ---------------------------------------------------------------------------
# Transformer Dataset
# ---------------------------------------------------------------------------

class FakeNewsNetDataset:
    """PyTorch Dataset for DistilBERT / BERT fine-tuning on FakeNewsNet."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: Any,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import torch
        text = clean_article_text(self.texts[idx], keep_case=True)[:2000]  # truncate before tokenizer
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def prepare_ml_features(
    train_df: Any,
    val_df: Any,
    test_df: Any,
    text_col: str = "text",
    label_col: str = "label",
    use_ner: bool = True,
    use_lemma: bool = True,
    max_features: int = 100_000,
    ngram_range: Tuple[int, int] = (1, 3),
) -> Dict[str, Any]:
    """
    Full feature preparation pipeline for traditional ML baselines.

    Returns dict with:
        X_train, y_train, X_val, y_val, X_test, y_test, vectorizer
    """
    from scipy.sparse import hstack, csr_matrix

    for df, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found in {name} DataFrame.")

    def _get_texts(df):
        if "title" in df.columns:
            return [combine_title_text(row) for _, row in df.iterrows()]
        return df[text_col].fillna("").tolist()

    train_texts = _get_texts(train_df)
    val_texts = _get_texts(val_df)
    test_texts = _get_texts(test_df)

    train_labels = train_df[label_col].tolist()
    val_labels = val_df[label_col].tolist()
    test_labels = test_df[label_col].tolist()

    # TF-IDF
    logger.info("Building TF-IDF features (use_lemma=%s) ...", use_lemma)
    X_train, _, vec = build_tfidf_features(
        train_texts, train_labels, fit=True,
        max_features=max_features, ngram_range=ngram_range, use_lemma=use_lemma,
    )
    X_val, _, _ = build_tfidf_features(
        val_texts, val_labels, vectorizer=vec, fit=False, use_lemma=use_lemma,
    )
    X_test, _, _ = build_tfidf_features(
        test_texts, test_labels, vectorizer=vec, fit=False, use_lemma=use_lemma,
    )

    # NER features
    if use_ner:
        logger.info("Extracting NER features ...")
        X_ner_train = build_ner_feature_matrix(train_texts)
        X_ner_val = build_ner_feature_matrix(val_texts)
        X_ner_test = build_ner_feature_matrix(test_texts)
        X_train = combine_tfidf_ner(X_train, X_ner_train)
        X_val = combine_tfidf_ner(X_val, X_ner_val)
        X_test = combine_tfidf_ner(X_test, X_ner_test)

    logger.info("FakeNewsNet features: train=%s, val=%s, test=%s",
                X_train.shape, X_val.shape, X_test.shape)

    return {
        "X_train": X_train, "y_train": train_labels,
        "X_val": X_val, "y_val": val_labels,
        "X_test": X_test, "y_test": test_labels,
        "vectorizer": vec,
    }

