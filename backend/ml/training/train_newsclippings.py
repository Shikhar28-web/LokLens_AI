"""
NewsCLIPpings Image-Text Consistency — Training Script
========================================================
Trains and compares on CLIP embeddings:
  1. Logistic Regression
  2. Random Forest
  3. XGBoost
  4. MLP (PyTorch)

Default mode: precomputed CLIP embeddings (Option A — no raw images needed).
GPU: RTX 4060 Laptop — MLP batch_size=256

Usage:
    python ml/training/train_newsclippings.py --mode compare
    python ml/training/train_newsclippings.py --mode mlp
    python ml/training/train_newsclippings.py --mode all
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

from ml.datasets.newsclippings_loader import load_all_splits, validate_dataset, clone_repo, run_download_script
from ml.preprocessing.newsclippings_preprocessor import build_feature_matrix, NewsCLIPDataset, ID2LABEL

logger = logging.getLogger(__name__)

SEED = 42
OUTPUT_DIR = Path("models/newsclippings_image_text")
NUM_LABELS = 2  # MATCHED, MISMATCHED
FUSION = "concat+diff+cos"


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
            return torch.device("cuda")
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Classical ML comparison
# ---------------------------------------------------------------------------

def train_classical(X_train, y_train, X_val, y_val, X_test, y_test, output_dir: Path) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix,
    )

    logger.info("=== NewsCLIPpings: Classical ML on CLIP embeddings ===")

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs", class_weight="balanced",
            random_state=SEED
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=SEED, n_jobs=-1, max_features="sqrt",
        ),
    }

    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=SEED, n_jobs=-1, tree_method="hist",
        )
    except ImportError:
        logger.warning("XGBoost not installed; skipping in classical comparison.")

    best_model, best_roc, best_name = None, 0.0, ""
    results = {}

    for name, model in models.items():
        logger.info("Training %s ...", name)
        try:
            model.fit(X_train, y_train)
            val_proba = model.predict_proba(X_val)[:, 1]
            val_preds = model.predict(X_val)
            val_roc = roc_auc_score(y_val, val_proba)
            val_f1 = f1_score(y_val, val_preds, average="weighted")
            val_acc = accuracy_score(y_val, val_preds)
            logger.info("[%s] Val acc=%.4f | ROC-AUC=%.4f | weighted F1=%.4f",
                        name, val_acc, val_roc, val_f1)
            results[name] = {"val_accuracy": val_acc, "val_roc_auc": val_roc, "val_f1_weighted": val_f1}

            if val_roc > best_roc:
                best_roc = val_roc
                best_model = (name, model)
                best_name = name
        except Exception as e:
            logger.error("Training %s failed: %s", name, e)

    logger.info("Best classical model: %s (val ROC-AUC=%.4f)", best_name, best_roc)

    best_name_s, best_clf = best_model
    test_preds = best_clf.predict(X_test)
    test_proba = best_clf.predict_proba(X_test)[:, 1]
    test_acc = accuracy_score(y_test, test_preds)
    test_f1_w = f1_score(y_test, test_preds, average="weighted")
    test_f1_m = f1_score(y_test, test_preds, average="macro")
    test_roc = roc_auc_score(y_test, test_proba)
    report = classification_report(y_test, test_preds, target_names=list(ID2LABEL.values()))
    cm = confusion_matrix(y_test, test_preds).tolist()

    logger.info("=== NEWSCLIPPINGS CLASSICAL TEST ===\n%s", report)

    results["best_model"] = best_name_s
    results["test_accuracy"] = test_acc
    results["test_f1_weighted"] = test_f1_w
    results["test_f1_macro"] = test_f1_m
    results["test_roc_auc"] = test_roc
    results["confusion_matrix"] = cm

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "classical_model.pkl", "wb") as f:
        pickle.dump(best_clf, f)
    with open(output_dir / "classical_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# MLP Classifier (PyTorch)
# ---------------------------------------------------------------------------

class MLP(object.__class__):
    pass


import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: list = None, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        hidden = hidden or [512, 256]
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test, output_dir: Path) -> dict:
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

    EPOCHS = 30
    BATCH_SIZE = 256
    LR = 1e-3
    PATIENCE = 5

    device = get_device()
    input_dim = X_train.shape[1]
    logger.info("=== NewsCLIPpings MLP (input_dim=%d, device=%s) ===", input_dim, device)

    train_ds = NewsCLIPDataset(X_train, y_train)
    val_ds = NewsCLIPDataset(X_val, y_val)
    test_ds = NewsCLIPDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    model = MLPClassifier(input_dim, hidden=[512, 256], num_classes=NUM_LABELS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_f1, patience_ctr = 0.0, 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            out = model(X_b.to(device))
            loss = criterion(out, y_b.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        all_preds, all_labels, all_proba = [], [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                out = model(X_b.to(device))
                proba = torch.softmax(out, dim=-1)[:, 1].cpu().numpy()
                preds = out.argmax(-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_b.numpy())
                all_proba.extend(proba)

        val_f1 = f1_score(all_labels, all_preds, average="weighted")
        val_acc = accuracy_score(all_labels, all_preds)
        scheduler.step(1 - val_f1)
        logger.info("[Epoch %d/%d] loss=%.4f | val_acc=%.4f | val_f1=%.4f",
                    epoch, EPOCHS, avg_loss, val_acc, val_f1)
        history.append({"epoch": epoch, "loss": avg_loss, "val_acc": val_acc, "val_f1": val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_ctr = 0
            torch.save(model.state_dict(), output_dir / "mlp_best.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Test
    model.load_state_dict(torch.load(output_dir / "mlp_best.pt"))
    model.eval()
    all_preds, all_labels, all_proba = [], [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            out = model(X_b.to(device))
            proba = torch.softmax(out, -1)[:, 1].cpu().numpy()
            preds = out.argmax(-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_b.numpy())
            all_proba.extend(proba)

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1_w = f1_score(all_labels, all_preds, average="weighted")
    test_f1_m = f1_score(all_labels, all_preds, average="macro")
    test_roc = roc_auc_score(all_labels, all_proba)
    report = classification_report(all_labels, all_preds, target_names=list(ID2LABEL.values()))
    logger.info("=== NEWSCLIPPINGS MLP TEST ===\n%s", report)

    results = {
        "model": "MLP", "best_val_f1": best_val_f1,
        "test_accuracy": test_acc,
        "test_f1_weighted": test_f1_w,
        "test_f1_macro": test_f1_m,
        "test_roc_auc": test_roc,
        "input_dim": input_dim,
        "training_history": history,
    }
    with open(output_dir / "mlp_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save model config for inference
    model_config = {"input_dim": input_dim, "hidden": [512, 256], "num_classes": NUM_LABELS}
    with open(output_dir / "mlp_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

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
    parser.add_argument("--mode", choices=["compare", "mlp", "all"], default="all")
    parser.add_argument("--setup", action="store_true",
                        help="Clone repo and run download.sh before training")
    args = parser.parse_args()

    set_seed(SEED)

    # Optionally set up dataset
    if args.setup:
        import yaml
        cfg_path = Path("ml/config/datasets.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        local_dir = cfg["datasets"]["newsclippings"]["local_dir"]
        clone_repo(local_dir)
        run_download_script(local_dir)

    # Validate
    for split in ["train", "val", "test"]:
        s = validate_dataset(split)
        logger.info("NewsCLIPpings %s: %s", split, s)

    # Load embeddings
    logger.info("Loading precomputed embeddings ...")
    result = load_all_splits(use_precomputed=True, fusion=FUSION)
    X_train, y_train, X_val, y_val, X_test, y_test = result

    # Apply scaling via build_feature_matrix already done inside loader
    logger.info("Feature shapes: train=%s, val=%s, test=%s",
                X_train.shape, X_val.shape, X_test.shape)

    all_results = {}
    if args.mode in ("compare", "all"):
        all_results["classical"] = train_classical(X_train, y_train, X_val, y_val, X_test, y_test, OUTPUT_DIR)
    if args.mode in ("mlp", "all"):
        all_results["mlp"] = train_mlp(X_train, y_train, X_val, y_val, X_test, y_test, OUTPUT_DIR)

    # Save label config
    with open(OUTPUT_DIR / "label_config.json", "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": {v: k for k, v in ID2LABEL.items()}}, f, indent=2)

    with open(OUTPUT_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("NewsCLIPpings training complete.")


if __name__ == "__main__":
    main()

