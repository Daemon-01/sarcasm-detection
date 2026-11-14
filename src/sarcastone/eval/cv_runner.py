"""Cross-validation orchestrator for SarcasTone experiments.

Implements the unified experimental protocol specified in AGENTS.md Section 7:
- Speaker-independent 5-fold cross validation.
- Multi-seed execution (default seeds: [42, 123, 2026]).
- Consistent result reporting with fold-level and seed-level tracking.
- Zero test-information leakage.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from pathlib import Path

import numpy as np
import pandas as pd

from sarcastone.data.folds import get_fold_split, load_frozen_folds
from sarcastone.eval.metrics import aggregate_fold_metrics
from sarcastone.utils.seeds import EXPERIMENT_SEEDS, set_seed

logger = logging.getLogger(__name__)


def run_cross_validation(
    manifest_df: pd.DataFrame,
    folds_data: Dict[str, Any],
    train_eval_fn: Callable[[pd.DataFrame, pd.DataFrame, int, int], Dict[str, Any]],
    seeds: Sequence[int] = EXPERIMENT_SEEDS,
    experiment_name: str = "experiment",
) -> Dict[str, Any]:
    """Execute multi-seed cross-validation across all frozen folds.

    Args:
        manifest_df: Full target manifest DataFrame.
        folds_data: Loaded frozen folds dictionary (data/folds.json).
        train_eval_fn: Callable taking (train_df, test_df, fold_idx, seed)
                       and returning a dict containing 'metrics' and optional 'predictions'.
        seeds: Random seeds to evaluate.
        experiment_name: Name identifier for the experiment.

    Returns:
        Structured cross-validation summary dictionary.
    """
    all_seed_results: List[Dict[str, Any]] = []

    for seed in seeds:
        set_seed(seed)
        fold_results: List[Dict[str, Any]] = []

        for fold in folds_data["folds"]:
            fold_idx = fold["fold"]
            train_df, test_df = get_fold_split(manifest_df, folds_data, fold_idx=fold_idx)

            res = train_eval_fn(train_df, test_df, fold_idx, seed)
            fold_record = {
                "fold": fold_idx,
                "seed": seed,
                "test_sample_count": len(test_df),
                "test_speakers": fold["test_speakers"],
                "metrics": res.get("metrics", {}),
            }
            if "predictions" in res:
                fold_record["predictions"] = res["predictions"]
            fold_results.append(fold_record)

        agg = aggregate_fold_metrics([fr["metrics"] for fr in fold_results])
        all_seed_results.append({
            "seed": seed,
            "aggregate_metrics": agg,
            "fold_results": fold_results,
        })

    # Summary across all seeds
    acc_means = [sr["aggregate_metrics"]["accuracy_mean"] for sr in all_seed_results]
    f1_macro_means = [sr["aggregate_metrics"]["f1_macro_mean"] for sr in all_seed_results]
    f1_weighted_means = [sr["aggregate_metrics"]["f1_weighted_mean"] for sr in all_seed_results]

    summary = {
        "experiment_name": experiment_name,
        "seeds": list(seeds),
        "accuracy_mean": round(float(np.mean(acc_means)), 4),
        "accuracy_std": round(float(np.std(acc_means)), 4),
        "f1_macro_mean": round(float(np.mean(f1_macro_means)), 4),
        "f1_macro_std": round(float(np.std(f1_macro_means)), 4),
        "f1_weighted_mean": round(float(np.mean(f1_weighted_means)), 4),
        "f1_weighted_std": round(float(np.std(f1_weighted_means)), 4),
        "seed_runs": all_seed_results,
    }

    logger.info(
        "Cross-validation '%s' finished: Accuracy = %.4f ± %.4f, Macro F1 = %.4f ± %.4f",
        experiment_name,
        summary["accuracy_mean"],
        summary["accuracy_std"],
        summary["f1_macro_mean"],
        summary["f1_macro_std"],
    )
    return summary

