"""
Validate pipeline structure for all 4 models without real data.
Run: .\venv\Scripts\python.exe ml\scripts\validate_pipelines.py
"""
import sys
import logging
sys.path.insert(0, '.')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

import numpy as np

# ── 1. FEVER preprocessor smoke test ──────────────────────────────────────────
logger.info("=== Testing FEVER preprocessor ===")
try:
    from ml.preprocessing.fever_preprocessor import (
        clean_text, build_tfidf_features, build_input_pair, normalize_label
    )
    # Synthetic pairs
    pairs = [("The sky is blue.", "The sky appears blue due to Rayleigh scattering."),
             ("Paris is in Germany.", "Paris is the capital of France.")]
    labels = [0, 1]
    X, y, vec = build_tfidf_features(pairs, labels, fit=True)
    logger.info("  FEVER TF-IDF shape: %s, labels: %s", X.shape, y)
    assert X.shape[0] == 2
    logger.info("  FEVER preprocessor OK")
except Exception as e:
    logger.error("  FEVER preprocessor FAILED: %s", e)

# ── 2. FakeNewsNet preprocessor smoke test ────────────────────────────────────
logger.info("=== Testing FakeNewsNet preprocessor ===")
try:
    from ml.preprocessing.fakenewsnet_preprocessor import (
        clean_article_text, build_tfidf_features as fnn_tfidf, ID2LABEL
    )
    texts = ["Breaking news: floods hit the city today.",
             "SHOCKING: Scientists discover cure for cancer (FAKE)"]
    labels = [0, 1]
    X, y, vec = fnn_tfidf(texts, labels, fit=True, max_features=500, ngram_range=(1, 2))
    logger.info("  FakeNewsNet TF-IDF shape: %s, labels: %s", X.shape, y)
    assert X.shape[0] == 2
    logger.info("  FakeNewsNet preprocessor OK")
except Exception as e:
    logger.error("  FakeNewsNet preprocessor FAILED: %s", e)

# ── 3. NewsCLIPpings preprocessor smoke test ─────────────────────────────────
logger.info("=== Testing NewsCLIPpings preprocessor ===")
try:
    from ml.preprocessing.newsclippings_preprocessor import build_feature_matrix, ID2LABEL as nc_labels
    # Synthetic embeddings (N=4, D=512)
    img_emb = np.random.randn(4, 512).astype(np.float32)
    txt_emb = np.random.randn(4, 512).astype(np.float32)
    for fusion in ["concat", "diff", "cosine", "concat+diff", "concat+diff+cos"]:
        X, scaler = build_feature_matrix(img_emb, txt_emb, fusion=fusion, fit=True)
        logger.info("  fusion='%s' → X.shape=%s", fusion, X.shape)
    logger.info("  NewsCLIPpings preprocessor OK")
except Exception as e:
    logger.error("  NewsCLIPpings preprocessor FAILED: %s", e)

# ── 4. FakeTT preprocessor smoke test ────────────────────────────────────────
logger.info("=== Testing FakeTT preprocessor ===")
try:
    from ml.preprocessing.fakett_preprocessor import (
        extract_metadata_features, ID2LABEL as ft_labels
    )
    sample = {"frames": ["a.jpg", "b.jpg"], "transcript": "Hello world", "description": "test", "hashtags": ["#news"]}
    meta = extract_metadata_features(sample)
    logger.info("  FakeTT metadata features: %s", meta)
    assert len(meta) == 5
    logger.info("  FakeTT preprocessor OK")
except Exception as e:
    logger.error("  FakeTT preprocessor FAILED: %s", e)

# ── 5. FakeTT dry-run end-to-end ──────────────────────────────────────────────
logger.info("=== FakeTT dry-run end-to-end feature fusion ===")
try:
    from ml.preprocessing.fakett_preprocessor import normalize_features, FakeTTDataset
    import torch
    X = np.random.randn(100, 901).astype(np.float32)
    y = np.random.randint(0, 3, 100).astype(np.int64)
    X_tr, X_val, X_te = X[:60], X[60:80], X[80:]
    y_tr, y_val, y_te = y[:60], y[60:80], y[80:]
    X_tr, X_val, X_te, scaler = normalize_features(X_tr, X_val, X_te)
    ds = FakeTTDataset(X_tr, y_tr)
    x0, y0 = ds[0]
    assert x0.shape == (901,), f"Expected (901,), got {x0.shape}"
    logger.info("  FakeTT feature matrix: train=%s, val=%s, test=%s", X_tr.shape, X_val.shape, X_te.shape)
    logger.info("  FakeTT dataset item: X=%s, y=%s", x0.shape, y0)
    logger.info("  FakeTT dry-run OK")
except Exception as e:
    logger.error("  FakeTT dry-run FAILED: %s", e)

# ── 6. Evaluation metrics smoke test ─────────────────────────────────────────
logger.info("=== Testing evaluation metrics ===")
try:
    from ml.evaluation.metrics import compute_classification_metrics, print_metrics
    y_true = [0, 1, 2, 0, 1, 1, 2, 0]
    y_pred = [0, 1, 1, 0, 0, 1, 2, 1]
    y_proba = np.random.dirichlet([1, 1, 1], size=8)
    metrics = compute_classification_metrics(y_true, y_pred, y_proba, label_names=["A", "B", "C"])
    print_metrics(metrics, model_name="Smoke Test")
    logger.info("  Evaluation metrics OK")
except Exception as e:
    logger.error("  Evaluation metrics FAILED: %s", e)

logger.info("=== All pipeline validations complete ===")

