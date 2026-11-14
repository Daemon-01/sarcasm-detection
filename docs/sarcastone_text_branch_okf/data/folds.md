---
type: concept
title: Shared cross-validation folds
description: Consumption contract for teammate-produced frozen folds.
tags: [cross-validation, folds, reproducibility]
---

# Shared folds

The text branch consumes the shared folds file produced by a teammate.

- Protocol: speaker-independent 5-fold cross-validation.
- Folds are frozen and must be reused unchanged.
- Do not generate replacement folds in the text branch.
- Run the agreed experiments with 3 seeds.

Implemented module: `src/sarcastone/data/folds.py`.

Frozen file: `data/folds.json`.

## Speaker-Independent 5-Fold Cross-Validation Specification
- **Protocol**: 5-fold cross-validation grouped strictly by `target_speaker` using `GroupKFold`.
- **Zero Speaker Leakage**: In every fold, the test speaker set is 100% disjoint from the train speaker set ($\text{train\_speakers} \cap \text{test\_speakers} = \emptyset$).
- **Sample Distribution**: Each fold contains nearly equal sample counts (~240 samples per fold: 239, 240, 239, 245, 239).
- **Major Character Distribution**: Naturally distributes the 5 largest characters (accounting for 55% of all samples) exactly one per fold:
  - **Fold 0 (239 samples)**: `CHANDLER` (157), `PERSON` (45), `PHOEBE` (35), `MEMBER-BOY` (2) — 59.0% sarcastic.
  - **Fold 1 (240 samples)**: `HOWARD` (137), `GILFOYLE` (46), `JOEY` (36), `DINESH` (8), `ERLICH` (4), `MEMBER-GIRL` (5), `RICHARD` (4) — 45.0% sarcastic.
  - **Fold 2 (239 samples)**: `PENNY` (127), `BERNADETTE` (53), `ROSS` (38), `STUART` (8), `MODERATOR` (7), `PERSON1` (4), `PERSON3` (1), `ROSE` (1) — 48.1% sarcastic.
  - **Fold 3 (245 samples)**: `SHELDON` (126), `AMY` (54), `OTHER` (36), `MONICA` (29) — 45.3% sarcastic.
  - **Fold 4 (239 samples)**: `LEONARD` (114), `RAJ` (55), `DOROTHY` (39), `RACHEL` (31) — 52.7% sarcastic.
- **Frozen Contract**: Saved to `data/folds.json` and reused unchanged across all model comparisons (TF-IDF baseline, RoBERTa embeddings, multimodal fusion).

See [evaluation conventions](../evaluation/conventions.md) and [project overview](../project/overview.md).
