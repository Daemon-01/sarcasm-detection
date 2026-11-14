"""Incongruent subset evaluation for SarcasTone.

Evaluates model performance on the sentiment-incongruent subset:
utterances where literal sentiment conflicts with true communicative intent
(e.g., positive literal sentiment in a sarcastic utterance).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd

from sarcastone.eval.metrics import compute_classification_metrics

logger = logging.getLogger(__name__)


def evaluate_sentiment_incongruence(
    predictions_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Evaluate predictions on sentiment-incongruent vs congruent subsets.

    Incongruent subset:
    - Positive literal text sentiment (predicted_sentiment == 'positive') but true label is sarcastic (sarcasm == 1).
    - Or Negative literal text sentiment but non-sarcastic.

    Args:
        predictions_df: DataFrame containing ['utterance_id', 'y_true', 'y_pred'].
        manifest_df: DataFrame containing ['utterance_id', 'sarcasm', 'valence'].
        sentiment_df: DataFrame containing ['utterance_id', 'predicted_sentiment'].

    Returns:
        Dictionary containing metric breakdowns on incongruent and congruent subsets.
    """
    merged = predictions_df.merge(
        sentiment_df[["utterance_id", "predicted_sentiment"]],
        on="utterance_id",
        how="inner",
    ).merge(
        manifest_df[["utterance_id", "valence"]],
        on="utterance_id",
        how="inner",
    )

    # Incongruent: positive surface sentiment with sarcasm=1, or negative surface sentiment with sarcasm=0
    is_positive_sarcastic = (merged["predicted_sentiment"] == "positive") & (merged["y_true"] == 1)
    is_negative_nonsarcastic = (merged["predicted_sentiment"] == "negative") & (merged["y_true"] == 0)
    incongruent_mask = is_positive_sarcastic | is_negative_nonsarcastic

    incongruent_subset = merged[incongruent_mask]
    congruent_subset = merged[~incongruent_mask]

    def _eval(subset: pd.DataFrame, name: str) -> Dict[str, Any]:
        if len(subset) == 0:
            return {"sample_count": 0, "accuracy": None}
        y_true = subset["y_true"].values
        y_pred = subset["y_pred"].values
        acc = float(np.mean(y_true == y_pred))
        return {
            "name": name,
            "sample_count": len(subset),
            "accuracy": round(acc, 4),
            "error_rate": round(1.0 - acc, 4),
        }

    report = {
        "total_evaluated": len(merged),
        "incongruent_subset": _eval(incongruent_subset, "sentiment_incongruent"),
        "congruent_subset": _eval(congruent_subset, "sentiment_congruent"),
        "positive_sarcastic_count": int(is_positive_sarcastic.sum()),
        "negative_nonsarcastic_count": int(is_negative_nonsarcastic.sum()),
    }

    logger.info(
        "Subset eval: Incongruent Acc = %s (N=%d) vs Congruent Acc = %s (N=%d)",
        report["incongruent_subset"]["accuracy"],
        report["incongruent_subset"]["sample_count"],
        report["congruent_subset"]["accuracy"],
        report["congruent_subset"]["sample_count"],
    )
    return report

