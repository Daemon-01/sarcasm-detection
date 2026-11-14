---
type: concept
title: Evaluation conventions
description: Shared evaluation protocol for text-branch experiments.
tags: [evaluation, metrics, reproducibility]
---

# Evaluation conventions

Use the shared evaluation harness and metrics written by a teammate. Do not reimplement them in the text branch.

Required metrics:
- Accuracy
- Precision
- Recall
- F1

Use:
- MUStARD++ only.
- Speaker-independent 5-fold cross-validation (`data/folds.json`).
- The frozen shared folds file generated via `src/sarcastone/data/folds.py`.
- 3 seeds (seeds: 42, 123, 2026).
- Standard classification metrics: Accuracy, Macro-Precision, Macro-Recall, Macro-F1, and Weighted-F1.

Annotated valence and arousal may be used for validating models only; never use them as input features.

## Averaging Convention (Resolved 2026-11-14)

All multi-class metric averaging is **macro** (unweighted mean across both classes), reflecting the balanced 50/50 MUStARD++ label distribution. Weighted-F1 is also reported for completeness.

- **Precision, Recall, F1**: `average='macro'` (macro-averaged over both classes).
- **Weighted-F1**: `average='weighted'` for secondary reference.
- **Accuracy**: Standard (number of correct / total predictions).
- **ROC AUC**: `sklearn.metrics.roc_auc_score` with binary probabilities.

## Result-Reporting Schema

Each experiment produces a JSON report under `results/` with the following hierarchy:
- `metadata`: model, pooling, protocol, seeds, conditions, n_folds, total_samples, timestamp.
- `results[condition].overall_summary`: `accuracy_mean`, `accuracy_std`, `f1_macro_mean`, `f1_macro_std`, `f1_weighted_mean`, `f1_weighted_std` aggregated over seeds × folds.
- `results[condition].seed_runs[seed].aggregate_metrics`: per-seed averages over 5 folds.
- `results[condition].seed_runs[seed].fold_results[fold].metrics`: per-fold values.
- `results[condition].seed_runs[seed].predictions`: per-sample `utterance_id`, `y_true`, `y_pred`, `y_prob`.

Implemented in `src/sarcastone/eval/metrics.py` (`compute_classification_metrics`, `aggregate_fold_metrics`, `format_metrics_summary`).

See [shared folds](../data/folds.md) and [project overview](../project/overview.md).
