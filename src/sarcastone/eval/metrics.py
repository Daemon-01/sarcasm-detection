"""Standardized evaluation metrics for SarcasTone classification experiments.

Computes accuracy, macro/weighted precision, recall, F1, and confusion matrix
in accordance with AGENTS.md Section 7 and OKF conventions:
docs/sarcastone_text_branch_okf/evaluation/conventions.md
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: Union[Sequence[int], np.ndarray],
    y_pred: Union[Sequence[int], np.ndarray],
    y_prob: Optional[Union[Sequence[float], np.ndarray]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive classification evaluation metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Optional predicted positive class probabilities.

    Returns:
        Dictionary of computed metrics:
        - accuracy
        - precision_macro, recall_macro, f1_macro
        - precision_weighted, recall_weighted, f1_weighted
        - precision_binary, recall_binary, f1_binary
        - confusion_matrix: {'tn': int, 'fp': int, 'fn': int, 'tp': int}
        - roc_auc (if y_prob is provided)
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)})")

    acc = float(accuracy_score(y_t, y_p))

    # Macro averaging (unweighted mean across classes)
    p_macro = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    r_macro = float(recall_score(y_t, y_p, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_t, y_p, average="macro", zero_division=0))

    # Weighted averaging (support-weighted mean)
    p_weighted = float(precision_score(y_t, y_p, average="weighted", zero_division=0))
    r_weighted = float(recall_score(y_t, y_p, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_t, y_p, average="weighted", zero_division=0))

    # Binary metrics specifically for the sarcastic class (label 1)
    p_binary = float(precision_score(y_t, y_p, pos_label=1, zero_division=0))
    r_binary = float(recall_score(y_t, y_p, pos_label=1, zero_division=0))
    f1_binary = float(f1_score(y_t, y_p, pos_label=1, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    metrics: Dict[str, Any] = {
        "accuracy": round(acc, 4),
        "precision_macro": round(p_macro, 4),
        "recall_macro": round(r_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(p_weighted, 4),
        "recall_weighted": round(r_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "precision_sarcastic": round(p_binary, 4),
        "recall_sarcastic": round(r_binary, 4),
        "f1_sarcastic": round(f1_binary, 4),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "support": int(len(y_t)),
    }

    if y_prob is not None:
        try:
            y_pr = np.asarray(y_prob, dtype=np.float64)
            auc = float(roc_auc_score(y_t, y_pr))
            metrics["roc_auc"] = round(auc, 4)
        except Exception as e:
            logger.debug("ROC AUC computation skipped: %s", e)

    return metrics


def aggregate_fold_metrics(fold_metrics_list: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean and standard deviation across cross-validation folds.

    Args:
        fold_metrics_list: List of metric dictionaries from each fold.

    Returns:
        Dictionary with mean and std for each scalar metric.
    """
    if not fold_metrics_list:
        return {}

    keys_to_aggregate = [
        "accuracy", "f1_macro", "precision_macro", "recall_macro",
        "f1_weighted", "precision_weighted", "recall_weighted",
        "f1_sarcastic", "precision_sarcastic", "recall_sarcastic",
    ]
    if "roc_auc" in fold_metrics_list[0]:
        keys_to_aggregate.append("roc_auc")

    summary: Dict[str, Any] = {}
    for key in keys_to_aggregate:
        values = [fm[key] for fm in fold_metrics_list if key in fm]
        if values:
            summary[f"{key}_mean"] = round(float(np.mean(values)), 4)
            summary[f"{key}_std"] = round(float(np.std(values)), 4)

    return summary


def format_metrics_summary(metrics: Dict[str, Any]) -> str:
    """Format metrics dictionary into a readable summary string.

    Args:
        metrics: Dictionary of metric values.

    Returns:
        Formatted multi-line string.
    """
    lines = []
    if "accuracy_mean" in metrics:
        lines.append(f"Accuracy: {metrics['accuracy_mean']:.4f} ± {metrics.get('accuracy_std', 0.0):.4f}")
        lines.append(f"Macro F1: {metrics['f1_macro_mean']:.4f} ± {metrics.get('f1_macro_std', 0.0):.4f}")
        lines.append(f"Macro Precision: {metrics['precision_macro_mean']:.4f} ± {metrics.get('precision_macro_std', 0.0):.4f}")
        lines.append(f"Macro Recall:    {metrics['recall_macro_mean']:.4f} ± {metrics.get('recall_macro_std', 0.0):.4f}")
        lines.append(f"Weighted F1:     {metrics['f1_weighted_mean']:.4f} ± {metrics.get('f1_weighted_std', 0.0):.4f}")
    elif "accuracy" in metrics:
        lines.append(f"Accuracy:        {metrics['accuracy']:.4f}")
        lines.append(f"Macro F1:        {metrics['f1_macro']:.4f}")
        lines.append(f"Macro Precision: {metrics['precision_macro']:.4f}")
        lines.append(f"Macro Recall:    {metrics['recall_macro']:.4f}")
        lines.append(f"Weighted F1:     {metrics['f1_weighted']:.4f}")
        if "confusion_matrix" in metrics:
            cm = metrics["confusion_matrix"]
            lines.append(f"Confusion Matrix: [TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}]")
    return "\n".join(lines)

