---
type: concept
title: TF-IDF and logistic regression baseline
description: Classical text-only baseline.
tags: [baseline, tf-idf, logistic-regression]
---

# TF-IDF + logistic regression

Implemented module: `src/sarcastone/text/tfidf.py`.

Results artifact: `results/tfidf_baseline_results.json`.

## TF-IDF + Logistic Regression Baseline Specification
- **Feature Extraction**: `TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)`.
- **Leakage Prevention**: Vectorizer is fit strictly on the training fold (`X_train`) and transforms test fold (`X_test`).
- **Classifier**: `LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs')`.
- **Evaluation Protocol**: Speaker-independent 5-fold cross-validation (`data/folds.json`) evaluated across 3 seeds (42, 123, 2026).
- **Ablation Comparison**: Evaluated across both `utterance_only` and `utterance_plus_context` conditions.

## Measured Baseline Results (5-Fold CV Across 3 Seeds)

| Input Condition | Accuracy (Mean ± Std) | Macro F1 (Mean ± Std) | Weighted F1 (Mean ± Std) |
|---|---|---|---|
| **Utterance Only** | $0.5432 \pm 0.0000$ | $0.5398 \pm 0.0000$ | $0.5430 \pm 0.0000$ |
| **Utterance + Context** | $0.5659 \pm 0.0000$ | $0.5623 \pm 0.0000$ | $0.5650 \pm 0.0000$ |

*Finding*: Incorporating conversational context (` [TURN] ` serialized) yields a **+2.27% accuracy** and **+2.25% Macro F1** gain over target text alone on the speaker-independent benchmark.

See [context ablation](../experiments/context-ablation.md) and [evaluation conventions](../evaluation/conventions.md).
