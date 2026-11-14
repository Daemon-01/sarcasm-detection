"""TF-IDF and Logistic Regression baseline for SarcasTone text branch.

Implements the classical text baseline mandated in AGENTS.md Section 6 & 7 and
OKF docs/sarcastone_text_branch_okf/models/tfidf-logreg.md:
- Uses TfidfVectorizer + LogisticRegression.
- Strictly fits vectorizer only on training folds to prevent data leakage.
- Evaluates across the frozen speaker-independent 5 folds (data/folds.json).
- Supports both 'utterance_only' and 'utterance_plus_context' ablation conditions.
- Executes across the 3-seed protocol (seeds: 42, 123, 2026).
- Computes comprehensive classification metrics (accuracy, macro/weighted F1).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sarcastone.data.folds import get_fold_split, load_frozen_folds
from sarcastone.eval.metrics import aggregate_fold_metrics, compute_classification_metrics
from sarcastone.text.preprocess import (
    UTTERANCE_ONLY,
    UTTERANCE_PLUS_CONTEXT,
    VALID_CONDITIONS,
    process_manifest,
)
from sarcastone.utils.seeds import EXPERIMENT_SEEDS, set_seed

logger = logging.getLogger(__name__)


def build_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    sublinear_tf: bool = True,
    min_df: int = 2,
) -> TfidfVectorizer:
    """Create a configured TfidfVectorizer instance.

    Args:
        max_features: Maximum vocabulary size.
        ngram_range: Lower and upper boundary of n-grams (default: (1, 2)).
        sublinear_tf: Apply sublinear tf scaling (1 + log(tf)).
        min_df: Minimum document frequency threshold.

    Returns:
        Configured TfidfVectorizer.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
    )


def train_eval_tfidf_fold(
    train_texts: List[str],
    train_labels: np.ndarray,
    test_texts: List[str],
    test_labels: np.ndarray,
    seed: int = 42,
    C: float = 1.0,
    max_iter: int = 1000,
    vectorizer: Optional[TfidfVectorizer] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Train and evaluate TF-IDF + Logistic Regression on a single fold.

    CRITICAL LEAKAGE PREVENTION: The vectorizer is fit strictly on train_texts,
    then applied via transform() to test_texts.

    Args:
        train_texts: Training fold text strings.
        train_labels: Training fold binary labels (0 or 1).
        test_texts: Test fold text strings.
        test_labels: Test fold binary labels (0 or 1).
        seed: Random seed for LogisticRegression solver.
        C: Inverse regularization strength.
        max_iter: Maximum solver iterations.
        vectorizer: Optional custom vectorizer.

    Returns:
        Tuple of (metrics_dict, y_pred, y_prob).
    """
    set_seed(seed)

    vec = vectorizer if vectorizer is not None else build_tfidf_vectorizer()

    # Fit ONLY on train data to prevent data leakage
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)

    # Train classifier
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
        class_weight=None,
    )
    clf.fit(X_train, train_labels)

    # Predict
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = compute_classification_metrics(test_labels, y_pred, y_prob=y_prob)
    return metrics, y_pred, y_prob


def run_tfidf_cv_single_seed(
    manifest_df: pd.DataFrame,
    folds_data: Dict[str, Any],
    condition: str = UTTERANCE_ONLY,
    seed: int = 42,
    C: float = 1.0,
) -> Dict[str, Any]:
    """Run full 5-fold cross-validation for a single seed under a given condition.

    Args:
        manifest_df: Master sample manifest DataFrame.
        folds_data: Loaded fold assignments.
        condition: 'utterance_only' or 'utterance_plus_context'.
        seed: Random seed.
        C: Logistic regression regularization parameter.

    Returns:
        Dictionary containing fold results, aggregated metrics, and predictions.
    """
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"Unknown condition '{condition}'. Expected one of {sorted(VALID_CONDITIONS)}")

    set_seed(seed)

    fold_results: List[Dict[str, Any]] = []
    all_predictions: List[Dict[str, Any]] = []

    for fold in folds_data["folds"]:
        fold_idx = fold["fold"]
        train_df, test_df = get_fold_split(manifest_df, folds_data, fold_idx=fold_idx)

        # Preprocess text samples for train and test
        train_samples = process_manifest(train_df, condition=condition)
        test_samples = process_manifest(test_df, condition=condition)

        train_texts = [s.get_input_text(condition=condition) for s in train_samples]
        train_labels = np.array([s.label for s in train_samples], dtype=np.int64)

        test_texts = [s.get_input_text(condition=condition) for s in test_samples]
        test_labels = np.array([s.label for s in test_samples], dtype=np.int64)
        test_uids = [s.utterance_id for s in test_samples]

        metrics, y_pred, y_prob = train_eval_tfidf_fold(
            train_texts=train_texts,
            train_labels=train_labels,
            test_texts=test_texts,
            test_labels=test_labels,
            seed=seed,
            C=C,
        )

        fold_record = {
            "fold": fold_idx,
            "seed": seed,
            "condition": condition,
            "test_sample_count": len(test_labels),
            "test_speakers": fold["test_speakers"],
            "metrics": metrics,
        }
        fold_results.append(fold_record)

        for uid, y_t, y_p, prob in zip(test_uids, test_labels, y_pred, y_prob):
            all_predictions.append({
                "utterance_id": uid,
                "fold": fold_idx,
                "seed": seed,
                "condition": condition,
                "y_true": int(y_t),
                "y_pred": int(y_p),
                "y_prob": round(float(prob), 4),
            })

    # Aggregate metrics across the 5 folds
    agg_metrics = aggregate_fold_metrics([fr["metrics"] for fr in fold_results])

    return {
        "seed": seed,
        "condition": condition,
        "aggregate_metrics": agg_metrics,
        "fold_results": fold_results,
        "predictions": all_predictions,
    }


def run_tfidf_experiment(
    manifest_path: Union[str, Path] = "data/manifest.csv",
    folds_path: Union[str, Path] = "data/folds.json",
    seeds: Sequence[int] = EXPERIMENT_SEEDS,
    conditions: Sequence[str] = (UTTERANCE_ONLY, UTTERANCE_PLUS_CONTEXT),
    output_dir: Optional[Union[str, Path]] = "results",
) -> Dict[str, Any]:
    """Execute complete 3-seed × 5-fold TF-IDF baseline experiment across conditions.

    Evaluates both 'utterance_only' and 'utterance_plus_context' conditions,
    aggregates fold and seed variances, and optionally writes output JSON.

    Args:
        manifest_path: Path to manifest.csv.
        folds_path: Path to folds.json.
        seeds: Sequence of random seeds (default: [42, 123, 2026]).
        conditions: Sequence of conditions to evaluate.
        output_dir: Optional output directory for results.

    Returns:
        Structured experiment results dictionary.
    """
    manifest_df = pd.read_csv(manifest_path)
    folds_data = load_frozen_folds(folds_path, manifest_path)

    results_by_condition: Dict[str, Any] = {}

    for cond in conditions:
        logger.info("Running TF-IDF baseline under condition: '%s'", cond)
        seed_runs: List[Dict[str, Any]] = []

        for seed in seeds:
            logger.info("Executing 5-fold CV for seed: %d", seed)
            run = run_tfidf_cv_single_seed(
                manifest_df=manifest_df,
                folds_data=folds_data,
                condition=cond,
                seed=seed,
            )
            seed_runs.append(run)

        # Cross-seed aggregation of the 5-fold mean scores
        seed_accs = [sr["aggregate_metrics"]["accuracy_mean"] for sr in seed_runs]
        seed_f1s = [sr["aggregate_metrics"]["f1_macro_mean"] for sr in seed_runs]
        seed_f1_weighted = [sr["aggregate_metrics"]["f1_weighted_mean"] for sr in seed_runs]

        overall_summary = {
            "accuracy_mean": round(float(np.mean(seed_accs)), 4),
            "accuracy_std": round(float(np.std(seed_accs)), 4),
            "f1_macro_mean": round(float(np.mean(seed_f1s)), 4),
            "f1_macro_std": round(float(np.std(seed_f1s)), 4),
            "f1_weighted_mean": round(float(np.mean(seed_f1_weighted)), 4),
            "f1_weighted_std": round(float(np.std(seed_f1_weighted)), 4),
        }

        results_by_condition[cond] = {
            "overall_summary": overall_summary,
            "seed_runs": seed_runs,
        }

        logger.info(
            "Condition '%s' completed: Accuracy = %.4f ± %.4f, Macro F1 = %.4f ± %.4f",
            cond,
            overall_summary["accuracy_mean"],
            overall_summary["accuracy_std"],
            overall_summary["f1_macro_mean"],
            overall_summary["f1_macro_std"],
        )

    experiment_report = {
        "metadata": {
            "model": "TF-IDF + Logistic Regression Baseline",
            "protocol": "Speaker-Independent 5-Fold Cross-Validation",
            "seeds": list(seeds),
            "conditions": list(conditions),
            "n_folds": len(folds_data["folds"]),
            "total_samples": len(manifest_df),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "results": results_by_condition,
    }

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        res_file = out_p / "tfidf_baseline_results.json"
        with open(res_file, "w", encoding="utf-8") as f:
            json.dump(experiment_report, f, indent=2)
        logger.info("Saved baseline experiment report to %s", res_file)

    return experiment_report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    repo_root = Path(__file__).resolve().parents[3]
    m_path = repo_root / "data" / "manifest.csv"
    f_path = repo_root / "data" / "folds.json"
    r_dir = repo_root / "results"
    run_tfidf_experiment(manifest_path=m_path, folds_path=f_path, output_dir=r_dir)

