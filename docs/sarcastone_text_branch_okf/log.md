---
type: log
title: Change log
description: Dated changes to the text-branch OKF bundle.
tags: [history]
---

# Change log

## 2026-11-12
- Created the text-branch OKF bundle.
- Recorded MUStARD++ as the only dataset.
- Documented TF-IDF + logistic regression, frozen RoBERTa-base mean-pooled embeddings, and pre-trained 3-class sentiment probabilities.
- Documented utterance-only versus utterance-plus-context ablation.
- Recorded speaker-independent frozen 5-fold CV and 3 seeds.
- Recorded shared `utterance_id` key, evaluation harness, metrics and classifier-head conventions.
- Marked unresolved implementation and handoff details as TODO.
- Completed MUStARD++ dataset inspection (Phase 1): documented 6,041 total utterances, 1,202 scenes, 50/50 balanced sarcasm labels (601/601), 4,839 context turns.
- Discovered and documented dataset anomalies: corrupted KEY `'Disgust'` in row 2596 (scene `1_S11E03_067`), corrupted SHOW `'0'` in row 613 (scene `1_2837`), context numbering gaps in 4 scenes, and 1,315 context turns with speaker `'-'`.
- Implemented `src/sarcastone/data/manifest.py`: generates `data/manifest.csv` (1,202 samples) and `data/manifest_utterances.csv` (6,041 turns). Enforces Option A (JSON turns) and Option B (` [TURN] ` serialized text), integer turn sorting, gap preservation, and modality availability tracking.
- Added comprehensive unit test suite in `tests/test_manifest.py` (8 passing tests).
- Corrected OKF path references in `src/sarcastone/agents.md` from `docs/okf/` to `docs/sarcastone_text_branch_okf/`.
- Resolved virtual environment import resolution: enabled `include-system-site-packages = true` in `.venv/pyvenv.cfg`, created `.venv/Lib/site-packages/sarcastone.pth`, added `.env` (`PYTHONPATH=src`), and populated `requirements.txt`.
- Implemented `src/sarcastone/text/preprocess.py`: deterministic text normalization (casing and punctuation preserved, no destructive lowercasing/stemming/stopwords), Option A and Option B context serialization (` [TURN] `), `TextSample` abstraction, and batch manifest processor supporting `utterance_only` and `utterance_plus_context` conditions.
- Added unit tests in `tests/test_preprocess.py` (all 14 unit tests passing).
- Confirmed Option A architecture: RoBERTa-base remains frozen with attention-mask-aware average (mean) pooling over non-padding tokens, training only a small classification head. Updated `models/roberta.md` and `interfaces/text-branch-output.md`.

## 2026-11-14
- Evaluated `StratifiedGroupKFold` vs. `GroupKFold` for speaker-independent cross-validation. Selected Option 2 (`GroupKFold` on `target_speaker`) because it naturally balances fold sizes (~240 samples per fold), distributes the 5 major characters (Chandler, Howard, Penny, Sheldon, Leonard) exactly one per fold, avoids greedy Chandler-Howard clumping, and achieves 45%–59% sarcasm class balance.
- Implemented `src/sarcastone/data/folds.py`: generates and validates speaker-independent 5-fold cross-validation split, saving frozen assignments to `data/folds.json`.
- Implemented unit tests in `tests/test_fold_leakage.py` verifying 0 speaker leakage, exact dataset partitioning, class presence, and split reproducibility (all 20 unit tests passing).
- Updated `docs/sarcastone_text_branch_okf/data/folds.md` and `docs/sarcastone_text_branch_okf/evaluation/conventions.md`.
- Implemented `src/sarcastone/utils/seeds.py` (seeds: 42, 123, 2026) and `src/sarcastone/eval/metrics.py` (accuracy, macro/weighted precision, recall, F1, and confusion matrix).
- Implemented `src/sarcastone/text/tfidf.py` (TF-IDF + Logistic Regression baseline with strict train-only vectorizer fitting).
- Executed the baseline experiment across all 5 speaker-independent folds and 3 seeds. Measured results: Utterance-Only (Acc: 54.32%, Macro F1: 53.98%), Utterance + Context (Acc: 56.59%, Macro F1: 56.23%), confirming a +2.27% accuracy gain from context.
- Saved results to `results/tfidf_baseline_results.json`.
- Implemented unit test suites `tests/test_metrics.py` and `tests/test_tfidf.py` (all 27 unit tests passing).
- Implemented `src/sarcastone/text/embed.py` containing attention-mask-aware mean pooling ($e = \sum h_i m_i / \sum m_i$), frozen `roberta-base` feature extractor, trainable `SarcasmClassifierHead` MLP, feature export, and 5-fold CV runner across seeds [42, 123, 2026].
- Executed RoBERTa cross-validation experiment:
  - Utterance Only: Accuracy = 59.80% ± 0.44%, Macro F1 = 59.61% ± 0.37% (statistically significant gain over TF-IDF baseline of +5.48% accuracy, p = 1.0e-5).
  - Utterance + Context: Accuracy = 58.87% ± 0.35%, Macro F1 = 58.50% ± 0.25%.
- Exported all 1,202 768-dim text embeddings to `features/text/{utterance_id}.npy` and compressed archives `roberta_embeddings_utterance_only.npz` and `roberta_embeddings_utterance_plus_context.npz`.
- Implemented `src/sarcastone/text/sentiment.py` using pre-trained 3-class sentiment model (`cardiffnlp/twitter-roberta-base-sentiment-latest`).
- Exported sentiment features for all 1,202 samples to `features/emotion/{utterance_id}.npy`, `sentiment_probabilities.npz`, and `sentiment_probabilities.csv`.
- Validated sentiment probabilities against annotated MUStARD++ valence (validation-only), demonstrating positive correlation (+0.2182) and monotonic valence progression across negative (3.65), neutral (4.13), and positive (4.41) predicted classes.
- Implemented shared evaluation modules:
  - `src/sarcastone/eval/cv_runner.py`: multi-seed cross-validation engine.
  - `src/sarcastone/eval/subsets.py`: sentiment-incongruent subset evaluator.
  - `src/sarcastone/eval/stats.py`: paired statistical testing (t-test, Wilcoxon, Markdown comparison tables).
- Implemented CLI entry points:
  - `scripts/run_experiment.py`: unified CLI to reproduce baseline and neural experiments.
  - `scripts/extract_text.py`: CLI to extract RoBERTa and sentiment features.
  - `scripts/extract_features.py` and `scripts/extract_audio.py`: dispatcher and explicit stubs.
- Implemented minimal interface stubs in `src/sarcastone/audio/` and `src/sarcastone/fusion/` explicitly raising `NotImplementedError` per AGENTS.md Section 8.
- Added comprehensive unit test suites in `tests/test_embed.py`, `tests/test_sentiment.py`, `tests/test_feature_shapes.py`, and `tests/test_eval_stats.py`. Full test suite achieves 100% pass rate (39 unit tests passing).
- Text branch implementation is fully complete according to OKF and AGENTS.md.
