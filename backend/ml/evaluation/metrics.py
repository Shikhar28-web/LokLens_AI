"""
Shared Evaluation Utilities
=============================
Computes all required metrics for each model and saves reports.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_proba: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Compute full metric suite:
      - Accuracy
      - Precision (macro + weighted)
      - Recall (macro + weighted)
      - F1 (macro + weighted)
      - Confusion matrix
      - ROC-AUC (binary or macro OvR)
      - PR-AUC (binary only)
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        classification_report,
        confusion_matrix,
    )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    num_classes = len(np.unique(y_true))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=label_names,
            zero_division=0,
        ),
    }

    # ROC-AUC
    if y_proba is not None:
        try:
            if num_classes == 2:
                proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba_pos))
                metrics["pr_auc"] = float(average_precision_score(y_true, proba_pos))
            else:
                metrics["roc_auc_macro"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
        except Exception as e:
            logger.warning("ROC-AUC computation failed: %s", e)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Save a JSON-serializable version (drop classification_report string)
        save_dict = {k: v for k, v in metrics.items() if k != "classification_report"}
        save_dict["classification_report_text"] = metrics.get("classification_report", "")
        with open(output_path, "w") as f:
            json.dump(save_dict, f, indent=2)
        logger.info("Evaluation report saved → %s", output_path)

    return metrics


def print_metrics(metrics: Dict[str, Any], model_name: str = ""):
    header = f"=== Evaluation: {model_name} ===" if model_name else "=== Evaluation ==="
    print(header)
    print(f"  Accuracy        : {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision macro : {metrics.get('precision_macro', 0):.4f}")
    print(f"  Precision weighted: {metrics.get('precision_weighted', 0):.4f}")
    print(f"  Recall macro    : {metrics.get('recall_macro', 0):.4f}")
    print(f"  Recall weighted : {metrics.get('recall_weighted', 0):.4f}")
    print(f"  F1 macro        : {metrics.get('f1_macro', 0):.4f}")
    print(f"  F1 weighted     : {metrics.get('f1_weighted', 0):.4f}")
    if "roc_auc" in metrics:
        print(f"  ROC-AUC         : {metrics['roc_auc']:.4f}")
    if "pr_auc" in metrics:
        print(f"  PR-AUC          : {metrics['pr_auc']:.4f}")
    if "roc_auc_macro" in metrics:
        print(f"  ROC-AUC (macro) : {metrics['roc_auc_macro']:.4f}")
    if "classification_report" in metrics:
        print("\n" + metrics["classification_report"])

