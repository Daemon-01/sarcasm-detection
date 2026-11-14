"""Evaluation package for SarcasTone."""

from sarcastone.eval.cv_runner import run_cross_validation
from sarcastone.eval.metrics import (
    aggregate_fold_metrics,
    compute_classification_metrics,
    format_metrics_summary,
)
from sarcastone.eval.stats import compute_paired_comparison, format_comparison_table
from sarcastone.eval.subsets import evaluate_sentiment_incongruence

__all__ = [
    "compute_classification_metrics",
    "aggregate_fold_metrics",
    "format_metrics_summary",
    "run_cross_validation",
    "compute_paired_comparison",
    "format_comparison_table",
    "evaluate_sentiment_incongruence",
]

