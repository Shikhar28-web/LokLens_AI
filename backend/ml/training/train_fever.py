"""
FEVER Claim Verifier — Training Script
========================================
Trains two models:
  1. Baseline: TF-IDF + Logistic Regression / SVM
  2. Primary:  Fine-tuned cross-encoder/nli-deberta-v3-small

GPU: RTX 4060 Laptop (8 GB VRAM) — batch_size=16, gradient_accumulation=2
Seed: 42 (reproducible)

Usage:
    python ml/training/train_fever.py --mode baseline
    python ml/training/train_fever.py --mode transformer
    python ml/training/train_fever.py --mode all
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

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.datasets.fever_loader import load_all_splits
from ml.preprocessing.fever_preprocessor import (
    build_tfidf_features,
    prepare_datasets,
    FEVERDataset,
    ID2LABEL,
)

logger = logging.getLogger(__name__)

SEED = 42
OUTPUT_DIR = Path("models/fever_claim_verifier")
NUM_LABELS = 3  # SUPPORTS, REFUTES, NOT ENOUGH INFO


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
            logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
            return dev
    except ImportError:
        pass
    logger.info("Using CPU.")
    return "cpu"


# ---------------------------------------------------------------------------
# Baseline: TF-IDF + Classical ML
# ---------------------------------------------------------------------------

def train_baseline(train_ds, val_ds, test_ds, output_dir: Path):
    """
    Train TF-IDF + Logistic Regression / LinearSVC baseline.
    Saves vectorizer + best model.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import classification_report, accuracy_score

    logger.info("=== FEVER Baseline: TF-IDF + Classical ML ===")

    # Unpack (claims, evidences, labels) tuples returned when tokenizer=None
    train_data, val_data, test_data = prepare_datasets(train_ds, val_ds, test_ds, tokenizer=None)
    train_claims, train_ev, train_labels = train_data
    val_claims, val_ev, val_labels = val_data
    test_claims, test_ev, test_labels = test_data

    # If official test set has hidden gold labels (-1), use a split of val (which is 20k balanced) for test evaluation
    if all(lbl < 0 for lbl in test_labels):
        logger.info("Official test set is unlabeled (-1). Splitting balanced dev set (20k) into 10k val / 10k test.")
        half = len(val_labels) // 2
        test_claims, test_ev, test_labels = val_claims[half:], val_ev[half:], val_labels[half:]
        val_claims, val_ev, val_labels = val_claims[:half], val_ev[:half], val_labels[:half]

    pairs_train = list(zip(train_claims, train_ev))
    pairs_val = list(zip(val_claims, val_ev))
    pairs_test = list(zip(test_claims, test_ev))

    X_train, y_train, vec = build_tfidf_features(pairs_train, train_labels, fit=True)
    X_val, y_val, _ = build_tfidf_features(pairs_val, val_labels, vectorizer=vec, fit=False)
    X_test, y_test, _ = build_tfidf_features(pairs_test, test_labels, vectorizer=vec, fit=False)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
        "linear_svc": CalibratedClassifierCV(
            LinearSVC(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
        ),
    }

    best_model, best_acc, best_name = None, 0.0, ""
    results = {}

    for name, model in models.items():
        logger.info("Training %s ...", name)
        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)
        acc = accuracy_score(y_val, val_preds)
        report = classification_report(y_val, val_preds, target_names=list(ID2LABEL.values()))
        logger.info("[%s] Val accuracy: %.4f\n%s", name, acc, report)
        results[name] = {"val_accuracy": acc}

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    logger.info("Best baseline: %s (val_acc=%.4f)", best_name, best_acc)

    # Test evaluation
    test_preds = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_report = classification_report(y_test, test_preds, target_names=list(ID2LABEL.values()))
    logger.info("FEVER BASELINE TEST ACCURACY: %.4f\n%s", test_acc, test_report)
    results["best_model"] = best_name
    results["test_accuracy"] = test_acc

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "baseline_vectorizer.pkl", "wb") as f:
        pickle.dump(vec, f)
    with open(output_dir / "baseline_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Baseline saved to %s", output_dir)
    return results


# ---------------------------------------------------------------------------
# Primary: Fine-tuned NLI Transformer
# ---------------------------------------------------------------------------

def train_transformer(train_ds, val_ds, test_ds, output_dir: Path):
    """Fine-tune cross-encoder/nli-deberta-v3-small on FEVER NLI pairs."""
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    pretrained_path = Path("models/pretrained_nli_deberta")
    MODEL_NAME = str(pretrained_path) if pretrained_path.exists() else "cross-encoder/nli-deberta-v3-small"
    EPOCHS = 2
    BATCH_SIZE = 16
    GRAD_ACC = 2            # effective batch = 32
    LR = 2e-5
    MAX_LEN = 256
    WARMUP_RATIO = 0.1
    MAX_TRAIN_SAMPLES = 18_000

    device = get_device()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== FEVER Transformer: %s ===", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
    )

    # Align pretrained NLI head (0=contradiction, 1=entailment, 2=neutral)
    # with FEVER labels (0=SUPPORTS/entailment, 1=REFUTES/contradiction, 2=NOT ENOUGH INFO/neutral)
    if str(getattr(model.config, "id2label", {}).get(0, "")).lower() == "contradiction":
        with torch.no_grad():
            if hasattr(model, "classifier") and model.classifier.weight.shape[0] == 3:
                model.classifier.weight[[0, 1]] = model.classifier.weight[[1, 0]].clone()
                if model.classifier.bias is not None:
                    model.classifier.bias[[0, 1]] = model.classifier.bias[[1, 0]].clone()
                logger.info("Aligned pretrained NLI classifier head (swapped 0<->1) to match FEVER labels.")

    model.config.id2label = ID2LABEL
    model.config.label2id = {v: k for k, v in ID2LABEL.items()}
    model = model.to(device)

    # Check if test set labels are hidden (-1)
    # Datasets
    if hasattr(test_ds, "__getitem__") and "label" in test_ds.column_names and set(test_ds["label"]) == {-1}:
        logger.info("Official test set is unlabeled (-1). Splitting dev set into val and test.")
        half = len(val_ds) // 2
        test_ds = val_ds.select(range(half, len(val_ds)))
        val_ds = val_ds.select(range(half))

    train_dataset, val_dataset, test_dataset = prepare_datasets(
        train_ds, val_ds, test_ds, tokenizer=tokenizer, max_length=MAX_LEN
    )

    # Stratified cap on train set for fast, balanced GPU fine-tuning
    if len(train_dataset) > MAX_TRAIN_SAMPLES:
        from collections import defaultdict
        by_cls = defaultdict(list)
        for idx, lbl in enumerate(train_dataset.labels):
            by_cls[lbl].append(idx)
        per_cls = MAX_TRAIN_SAMPLES // NUM_LABELS
        rng = np.random.RandomState(SEED)
        sel = []
        for cls_id in sorted(by_cls.keys()):
            idxs = np.array(by_cls[cls_id])
            chosen = rng.choice(idxs, size=min(per_cls, len(idxs)), replace=False)
            sel.extend(chosen.tolist())
        rng.shuffle(sel)
        train_dataset.claims = [train_dataset.claims[i] for i in sel]
        train_dataset.evidences = [train_dataset.evidences[i] for i in sel]
        train_dataset.labels = [train_dataset.labels[i] for i in sel]
        logger.info("Stratified train subset: %d examples (%d per class)", len(train_dataset), per_cls)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 4, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 4, shuffle=False, num_workers=0, pin_memory=True)

    total_steps = len(train_loader) * EPOCHS // GRAD_ACC
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_f1, best_epoch = 0.0, 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        use_bf16 = (getattr(device, "type", str(device)) == "cuda")
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / GRAD_ACC
            loss.backward()
            total_loss += outputs.loss.item()

            if (step + 1) % GRAD_ACC == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)

        # --- Validate ---
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
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        logger.info("[Epoch %d/%d] loss=%.4f | val_acc=%.4f | val_macro_f1=%.4f",
                    epoch, EPOCHS, avg_loss, val_acc, val_f1)
        history.append({"epoch": epoch, "loss": avg_loss, "val_acc": val_acc, "val_f1": val_f1})

        # Save best checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")
            logger.info("  → Checkpoint saved (best val_f1=%.4f at epoch %d)", best_val_f1, best_epoch)

    # --- Test evaluation with best model ---
    logger.info("Loading best checkpoint from epoch %d ...", best_epoch)
    model = AutoModelForSequenceClassification.from_pretrained(output_dir / "best_model").to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds = out.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1_macro = f1_score(all_labels, all_preds, average="macro")
    test_f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    test_report = classification_report(
        all_labels, all_preds, target_names=list(ID2LABEL.values())
    )
    logger.info("=== FEVER TRANSFORMER TEST RESULTS ===")
    logger.info("Test accuracy : %.4f", test_acc)
    logger.info("Test macro F1 : %.4f", test_f1_macro)
    logger.info("Test weighted F1: %.4f", test_f1_weighted)
    logger.info("\n%s", test_report)

    results = {
        "model": MODEL_NAME,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1_macro,
        "test_f1_weighted": test_f1_weighted,
        "training_history": history,
    }
    with open(output_dir / "transformer_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save label mapping
    with open(output_dir / "label_config.json", "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": {v: k for k, v in ID2LABEL.items()}}, f, indent=2)

    logger.info("Transformer results saved to %s", output_dir)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "training.log", mode="w"),
        ]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description="Train FEVER Claim Verifier")
    parser.add_argument("--mode", choices=["baseline", "transformer", "all"], default="all",
                        help="Training mode: baseline | transformer | all")
    args = parser.parse_args()

    set_seed(SEED)
    logger.info("Loading FEVER dataset ...")
    train_ds, val_ds, test_ds = load_all_splits(use_nli=True)
    logger.info("Train: %d | Val: %d | Test: %d", len(train_ds), len(val_ds), len(test_ds))

    all_results = {}

    if args.mode in ("baseline", "all"):
        all_results["baseline"] = train_baseline(train_ds, val_ds, test_ds, OUTPUT_DIR)

    if args.mode in ("transformer", "all"):
        all_results["transformer"] = train_transformer(train_ds, val_ds, test_ds, OUTPUT_DIR)

    with open(OUTPUT_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("All FEVER training complete. Results: %s", OUTPUT_DIR / "all_results.json")


if __name__ == "__main__":
    main()

