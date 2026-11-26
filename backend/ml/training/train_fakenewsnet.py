"""
FakeNewsNet Classifier — Training Script
==========================================
Trains and compares:
  1. Logistic Regression (TF-IDF baseline)
  2. Naive Bayes (Multinomial)
  3. Random Forest
  4. XGBoost
  5. DistilBERT fine-tuned

Selects best model by weighted F1 on validation set.
GPU: RTX 4060 Laptop (8 GB VRAM) — DistilBERT batch_size=16

Usage:
    python ml/training/train_fakenewsnet.py --mode baseline
    python ml/training/train_fakenewsnet.py --mode transformer
    python ml/training/train_fakenewsnet.py --mode all
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.datasets.fakenewsnet_loader import load_dataset_auto, train_val_test_split, validate_dataset
from ml.preprocessing.fakenewsnet_preprocessor import (
    prepare_ml_features, FakeNewsNetDataset, ID2LABEL, clean_article_text, combine_title_text,
)

logger = logging.getLogger(__name__)

SEED = 42
OUTPUT_DIR = Path("models/fakenewsnet_classifier")
NUM_LABELS = 2  # REAL, FAKE


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            logger.info("GPU: %s", torch.cuda.get_device_name(0))
            return dev
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Baseline classifiers
# ---------------------------------------------------------------------------

def train_baseline(train_df, val_df, test_df, output_dir: Path) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        classification_report, roc_auc_score, confusion_matrix,
    )
    import scipy.sparse as sp

    logger.info("=== FakeNewsNet Baseline: TF-IDF + Classical ML ===")

    feats = prepare_ml_features(
        train_df, val_df, test_df,
        use_ner=True, use_lemma=False,
        max_features=50_000, ngram_range=(1, 2),
    )
    X_train, y_train = feats["X_train"], feats["y_train"]
    X_val, y_val = feats["X_val"], feats["y_val"]
    X_test, y_test = feats["X_test"], feats["y_test"]
    vectorizer = feats["vectorizer"]

    # Scipy sparse matrices are supported natively by LogisticRegression and MultinomialNB
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", class_weight="balanced",
            random_state=SEED
        ),
        "naive_bayes": MultinomialNB(alpha=0.1),
        "random_forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=SEED, n_jobs=-1
        ),
    }

    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5,
            eval_metric="logloss",
            random_state=SEED, n_jobs=-1,
        )
    except ImportError:
        logger.warning("XGBoost not installed; continuing with LR, NB, and Random Forest.")

    best_model, best_f1, best_name = None, 0.0, ""
    results = {}

    for name, model in models.items():
        logger.info("Training %s ...", name)
        try:
            model.fit(X_train, y_train)
            val_preds = model.predict(X_val)

            val_f1 = f1_score(y_val, val_preds, average="weighted")
            val_acc = accuracy_score(y_val, val_preds)
            logger.info("[%s] Val acc=%.4f | weighted F1=%.4f", name, val_acc, val_f1)
            results[name] = {"val_accuracy": val_acc, "val_f1_weighted": val_f1}

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model = (name, model)
                best_name = name
        except Exception as e:
            logger.error("Training %s failed: %s", name, e)

    logger.info("Best baseline: %s (val weighted F1=%.4f)", best_name, best_f1)

    # Test eval
    best_name_saved, best_clf = best_model
    test_preds = best_clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds, average="weighted")
    test_rec = recall_score(y_test, test_preds, average="weighted")
    test_f1_w = f1_score(y_test, test_preds, average="weighted")
    test_f1_m = f1_score(y_test, test_preds, average="macro")
    report = classification_report(y_test, test_preds, target_names=list(ID2LABEL.values()))
    cm = confusion_matrix(y_test, test_preds).tolist()

    logger.info("=== FAKENEWSNET BASELINE TEST ===\n%s", report)

    results["best_model"] = best_name_saved
    results["test_accuracy"] = test_acc
    results["test_precision"] = test_prec
    results["test_recall"] = test_rec
    results["test_f1_weighted"] = test_f1_w
    results["test_f1_macro"] = test_f1_m
    results["confusion_matrix"] = cm

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "baseline_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open(output_dir / "baseline_model.pkl", "wb") as f:
        pickle.dump(best_clf, f)
    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# DistilBERT Fine-Tuning
# ---------------------------------------------------------------------------

def train_transformer(train_df, val_df, test_df, output_dir: Path) -> dict:
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizerFast,
        get_linear_schedule_with_warmup,
    )
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    pretrained_path = Path("models/pretrained_distilbert")
    MODEL_NAME = str(pretrained_path) if pretrained_path.exists() else "distilbert-base-uncased"
    EPOCHS = 3
    BATCH_SIZE = 16
    LR = 2e-5
    MAX_LEN = 512
    WARMUP_RATIO = 0.1

    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== FakeNewsNet Transformer: %s ===", MODEL_NAME)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    ).to(device)

    def _make_texts(df):
        if "title" in df.columns:
            return [combine_title_text(row) for _, row in df.iterrows()]
        return df["text"].fillna("").tolist()

    train_texts = _make_texts(train_df)
    val_texts = _make_texts(val_df)
    test_texts = _make_texts(test_df)
    train_labels = train_df["label"].tolist()
    val_labels = val_df["label"].tolist()
    test_labels = test_df["label"].tolist()

    train_dataset = FakeNewsNetDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = FakeNewsNetDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    test_dataset = FakeNewsNetDataset(test_texts, test_labels, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_f1, best_epoch = 0.0, 0
    history = []

    use_bf16 = (getattr(device, "type", str(device)) == "cuda")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += outputs.loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    out = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                    )
                preds = out.logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["labels"].numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average="weighted")
        logger.info("[Epoch %d/%d] loss=%.4f | val_acc=%.4f | val_f1=%.4f",
                    epoch, EPOCHS, avg_loss, val_acc, val_f1)
        history.append({"epoch": epoch, "loss": avg_loss, "val_acc": val_acc, "val_f1": val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            model.config.id2label = ID2LABEL
            model.config.label2id = {v: k for k, v in ID2LABEL.items()}
            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")

    # Test
    from sklearn.metrics import precision_score, recall_score, confusion_matrix
    model = DistilBertForSequenceClassification.from_pretrained(output_dir / "best_model").to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            all_preds.extend(out.logits.argmax(-1).cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_prec = precision_score(all_labels, all_preds, average="weighted")
    test_rec = recall_score(all_labels, all_preds, average="weighted")
    test_f1_macro = f1_score(all_labels, all_preds, average="macro")
    test_f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(all_labels, all_preds, target_names=list(ID2LABEL.values()))
    cm = confusion_matrix(all_labels, all_preds).tolist()
    logger.info("=== FAKENEWSNET TRANSFORMER TEST ===\n%s", report)

    results = {
        "model": MODEL_NAME, "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_accuracy": test_acc,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "test_f1_macro": test_f1_macro,
        "test_f1_weighted": test_f1_weighted,
        "confusion_matrix": cm,
        "training_history": history,
    }
    with open(output_dir / "transformer_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "label_config.json", "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": {v: k for k, v in ID2LABEL.items()}}, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "training.log", mode="w"),
        ]
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "transformer", "all"], default="all")
    args = parser.parse_args()

    set_seed(SEED)

    logger.info("Loading FakeNewsNet dataset ...")
    df = load_dataset_auto()
    summary = validate_dataset(df)
    logger.info("Dataset summary: %s", summary)

    train_df, val_df, test_df = train_val_test_split(df, seed=SEED)

    all_results = {}
    if args.mode in ("baseline", "all"):
        all_results["baseline"] = train_baseline(train_df, val_df, test_df, OUTPUT_DIR)
    if args.mode in ("transformer", "all"):
        all_results["transformer"] = train_transformer(train_df, val_df, test_df, OUTPUT_DIR)

    with open(OUTPUT_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("FakeNewsNet training complete.")


if __name__ == "__main__":
    main()

