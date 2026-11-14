"""Statistical testing and comparative analysis for SarcasTone.

Computes paired statistical tests and formatting tables between baseline
and neural models across the 5-fold cross-validation runs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def compute_paired_comparison(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    metric_name: str = "accuracy",
) -> Dict[str, Any]:
    """Compute paired t-test and Wilcoxon test between two models across folds.

    Args:
        scores_a: Fold scores from model A.
        scores_b: Fold scores from model B.
        model_a_name: Label for model A.
        model_b_name: Label for model B.
        metric_name: Metric being evaluated.

    Returns:
        Dictionary containing mean, std, mean difference, t-statistic, and p-value.
    """
    arr_a = np.asarray(scores_a, dtype=np.float64)
    arr_b = np.asarray(scores_b, dtype=np.float64)

    if len(arr_a) != len(arr_b):
        raise ValueError(f"Score lengths must match ({len(arr_a)} vs {len(arr_b)})")

    diff = arr_b - arr_a
    t_stat, p_val = stats.ttest_rel(arr_b, arr_a)

    return {
        "metric": metric_name,
        "model_a": {
            "name": model_a_name,
            "mean": round(float(np.mean(arr_a)), 4),
            "std": round(float(np.std(arr_a)), 4),
        },
        "model_b": {
            "name": model_b_name,
            "mean": round(float(np.mean(arr_b)), 4),
            "std": round(float(np.std(arr_b)), 4),
        },
        "mean_difference": round(float(np.mean(diff)), 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_val), 6),
        "statistically_significant_p05": bool(p_val < 0.05),
    }


def format_comparison_table(comparisons: List[Dict[str, Any]]) -> str:
    """Format comparisons into a GitHub-style markdown table."""
    if not comparisons:
        return ""
    name_a = comparisons[0]["model_a"]["name"]
    name_b = comparisons[0]["model_b"]["name"]
    lines = [
        f"| Metric | {name_a} | {name_b} | Diff ({name_b} - {name_a}) | p-value | Significant (p < 0.05) |",
        "|---|---|---|---|---|---|",
    ]
    for comp in comparisons:
        m_a = f"{comp['model_a']['mean']:.4f} ± {comp['model_a']['std']:.4f}"
        m_b = f"{comp['model_b']['mean']:.4f} ± {comp['model_b']['std']:.4f}"
        diff_str = f"{comp['mean_difference']:+.4f}"
        sig_str = "Yes" if comp["statistically_significant_p05"] else "No"
        lines.append(f"| {comp['metric']} | {m_a} | {m_b} | {diff_str} | {comp['p_value']:.4f} | {sig_str} |")

    return "\n".join(lines)
