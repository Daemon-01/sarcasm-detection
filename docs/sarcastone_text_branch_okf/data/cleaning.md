---
type: concept
title: Cleaning and sample construction
description: Contract for constructing text-branch samples.
tags: [cleaning, samples]
---

# Cleaning and sample construction

Implemented modules: `src/sarcastone/data/manifest.py` and `src/sarcastone/text/preprocess.py`.

**Responsibility:** apply the agreed text-data cleaning policy and construct standardized text-branch samples per `utterance_id`.

## Cleaning & Sample Construction Invariants
- **Sample Contract**: Each sample provides `target_text`, structured `context_turns` (Option A), and serialized `context_text` (Option B).
- **Ablation Support**: Supports both `utterance_only` and `utterance_plus_context` input modes via `TextSample.get_input_text()`.
- **Target Association**: Target utterances are identified by non-null `Sarcasm` (the final turn of each scene), while preceding turns are context turns.
- **Normalization Policy**: Whitespace and line breaks are normalized deterministically. Forced lowercasing, stemming, lemmatization, and stop-word deletion are strictly prohibited.
- **Anomaly Provenance**: Known dataset anomalies (row 2596 `KEY = 'Disgust'`, row 613 `SHOW = '0'`) are resolved deterministically while preserving original raw values.

See [context representation](context.md), [validation](validation.md) and [target text](../text/target-text.md).
