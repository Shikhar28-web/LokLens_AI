# LokLens AI — ML Training Pipeline Documentation

## Overview

This document explains how to set up, train, and run the ML models for Text and Image verification in LokLens AI:
1. **FEVER Claim Verification**: Natural Language Inference (SUPPORTS, REFUTES, NOT ENOUGH INFO)
2. **FakeNewsNet News Classifier**: Article text & headline classification (FAKE, REAL)
3. **NewsCLIPpings Consistency**: Multimodal Image-Text consistency (MATCHED, MISMATCHED)

---

## Directory Structure

```
ml/
├── config/
│   └── datasets.yaml          ← Edit dataset paths here (never in code)
├── datasets/
│   ├── fever_loader.py        ← FEVER HuggingFace loader
│   ├── fakenewsnet_loader.py  ← FakeNewsNet loader
│   └── newsclippings_loader.py← NewsCLIPpings annotation + embedding loader
├── preprocessing/
│   ├── fever_preprocessor.py
│   ├── fakenewsnet_preprocessor.py
│   └── newsclippings_preprocessor.py
├── training/
│   ├── train_fever.py
│   ├── train_fakenewsnet.py
│   └── train_newsclippings.py
├── evaluation/
│   └── metrics.py             ← Shared evaluation utilities
└── inference/
    ├── fever_inference.py
    ├── fakenewsnet_inference.py
    ├── newsclippings_inference.py
    └── unified_router.py      ← Route inputs to text / image model(s)

models/
├── fever_claim_verifier/
├── fakenewsnet_classifier/
└── newsclippings_image_text/
```

---

## Setup

### 1. Install Additional Dependencies

```bash
cd d:\Projects\Political_lens\backend
.\venv\Scripts\pip.exe install -r requirements-ml.txt
```

### 2. Configure Dataset Paths

Edit `ml/config/datasets.yaml` if your dataset files are in non-default locations.
Never hardcode paths in training scripts.

---

## Dataset Setup

### 1. FEVER (`pietrolesci/nli_fever` — Automatic)

Downloads automatically from HuggingFace into `data/datasets/fever/` (`208,346` train / `19,998` val / `19,998` test).
`fever_preprocessor.py` maps `premise` (claim) and `hypothesis` (evidence) and aligns labels to `0 = SUPPORTS`, `1 = REFUTES`, `2 = NOT ENOUGH INFO`.

```bash
.\venv\Scripts\python.exe ml/datasets/fever_loader.py
```

### 2. FakeNewsNet + ISOT Political News (Local / Kaggle)

- **FakeNewsNet content CSVs** (`data/datasets/fakenewsnet/`): `BuzzFeed_{fake,real}_news_content.csv` and `PolitiFact_{fake,real}_news_content.csv`.
- **ISOT Political News** (`data/datasets/isot/`): `Fake.csv` (`politics`, `Government News`) and `True.csv` (`politicsNews`).
- `fakenewsnet_loader.py` merges, deduplicates conflicting titles, and balances classes (`2,532` total: `1,266` REAL / `1,266` FAKE).
- `fakenewsnet_preprocessor.py` strips wire-service datelines (`WASHINGTON (Reuters) -`, `BRUSSELS —`, `(AP)`, `(AFP)`) to eliminate publisher-tag shortcut bias.

### 3. NewsCLIPpings (`merged_balanced` — Semi-Automatic)

Option A (default — precomputed `openai/clip-vit-base-patch32` embeddings, `71,072` train / `7,024` val / `7,264` test):
```bash
# Clone repo and download annotations + precomputed embeddings
.\venv\Scripts\python.exe ml/training/train_newsclippings.py --setup --mode all
```
No raw images needed. The `--setup` flag clones the GitHub repo and runs `download.sh`.

Option B (raw images — requires VisualNews access):
1. Submit request at the [VisualNews GitHub](https://github.com/g-luo/news_clippings)
2. Extract downloaded files to `data/datasets/visual_news/`
3. Set `use_precomputed_embeddings: false` in `datasets.yaml`

---

## Training & Current Benchmarks

All training scripts support GPU automatically (RTX 4060 — CUDA 12.6, `bfloat16` mixed precision).

### FEVER Claim Verifier (`cross-encoder/nli-deberta-v3-small`)

- **Model showing `74.85%` accuracy**: Fine-tuned **`cross-encoder/nli-deberta-v3-small`** (`models/fever_claim_verifier/best_model/`, `model_type: "transformer"`). *(The TF-IDF + Logistic Regression baseline scored `29.54%`.)*
- **Current Test Metrics (9,999 pairs)**: Accuracy **`74.85%`**, Macro/Weighted F1 **`0.7484`**
- **Single Input Format (`POST /api/ml/verify-claim`)**:
  ```json
  {
    "claim": "The BJP won 240 seats in India's 2024 parliamentary election.",
    "evidence": "The Bharatiya Janata Party won 240 seats in India's 2024 Lok Sabha election."
  }
  ```

```bash
# Baseline only (TF-IDF + Logistic Regression)
.\venv\Scripts\python.exe ml/training/train_fever.py --mode baseline

# Transformer fine-tuning (cross-encoder/nli-deberta-v3-small, 18k stratified subset)
.\venv\Scripts\python.exe ml/training/train_fever.py --mode transformer

# Both
.\venv\Scripts\python.exe ml/training/train_fever.py --mode all
```

### FakeNewsNet Political Classifier (`distilbert-base-uncased` + `XGBoost`)

- **Current Test Metrics (254 unseen articles, dateline-stripped)**:
  - **DistilBERT (`transformer`)**: Accuracy **`93.31%`**, F1 **`0.9331`**, Confusion Matrix `[[118, 9], [8, 119]]`
  - **XGBoost (`baseline`)**: Accuracy **`92.52%`**, F1 **`0.9252`**, Confusion Matrix `[[119, 8], [11, 116]]`

```bash
# Traditional ML baselines (LR / NB / RF / XGBoost with TF-IDF + NER)
.\venv\Scripts\python.exe ml/training/train_fakenewsnet.py --mode baseline

# DistilBERT fine-tuning
.\venv\Scripts\python.exe ml/training/train_fakenewsnet.py --mode transformer

# Both
.\venv\Scripts\python.exe ml/training/train_fakenewsnet.py --mode all
```

### NewsCLIPpings Image-Text Consistency (`PyTorch MLP` + `XGBoost` on CLIP)

- **Current Test Metrics (7,264 pairs)**:
  - **PyTorch MLP (`[512, 256]`)**: Accuracy **`72.16%`**, F1 **`0.7214`**, ROC-AUC **`0.7991`**
  - **XGBoost (`classical`)**: Accuracy **`69.62%`**, F1 **`0.6959`**, ROC-AUC **`0.7776`**

```bash
# Classical ML (LR / RF / XGB) on 1,537-d fused CLIP embeddings
.\venv\Scripts\python.exe ml/training/train_newsclippings.py --mode compare

# MLP classifier
.\venv\Scripts\python.exe ml/training/train_newsclippings.py --mode mlp

# Both (with dataset setup)
.\venv\Scripts\python.exe ml/training/train_newsclippings.py --setup --mode all
```

---

## Inference

### Individual model inference

```python
from ml.inference.fever_inference import predict as fever_predict
result = fever_predict(claim="The moon landing happened in 1969.", evidence="Apollo 11 landed on the Moon on July 20, 1969.")
print(result)

from ml.inference.fakenewsnet_inference import predict as fnn_predict
result = fnn_predict(text="..article text..", title="Headline")
print(result)

from ml.inference.newsclippings_inference import predict as nc_predict
result = nc_predict(image_path="path/to/image.jpg", caption="Image caption text")
print(result)
```

### Unified router (recommended for integration)

```python
from ml.inference.unified_router import router

# Text analysis
result = router.analyze_text(text="...", title="...", claims=["claim1"], evidence_texts=["evidence1"])

# Image analysis
result = router.analyze_image(image_path="img.jpg", caption="caption text")
```

---

## Model Outputs

All inference functions return structured dicts. Confidence scores reflect the
**model's output distribution**, NOT the probability that a news claim is factually true.
Always combine ML outputs with evidence retrieval results before presenting verdicts.

---

## Reproducibility

All training uses `SEED=42`. GPU training results may vary slightly between runs due to
CUDA non-determinism. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` for full determinism at the
cost of some performance.

---

## Hardware Notes

- **GPU**: NVIDIA RTX 4060 Laptop (8 GB VRAM) — CUDA 12.6
- **Batch sizes** are tuned for 8 GB VRAM. If you see OOM errors, halve the batch size.
- **Transformer fine-tuning**: FEVER ~3h, FakeNewsNet ~1h (all with RTX 4060)
- **XGBoost** uses GPU acceleration (`device="cuda"`) automatically
- **Whisper** uses GPU for transcription when available

