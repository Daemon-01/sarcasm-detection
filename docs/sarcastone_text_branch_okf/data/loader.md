---
type: concept
title: Loader
description: Contract for the text data-loading module.
tags: [loader, data-pipeline]
---

# Loader

Implemented module: `src/sarcastone/data/manifest.py`.

**Responsibility:** read the raw MUStARD++ dataset (`data/mustard++_text.csv`), validate records, and generate standardized manifests:
- Master sample-level manifest: `data/manifest.csv` (1,202 rows).
- Utterance-level registry: `data/manifest_utterances.csv` (6,041 rows).

**Preserved Fields**
- `utterance_id` (`SCENE` e.g. `1_10004`) for every target sample.
- `target_text`: exact text of the target utterance (casing, punctuation preserved).
- `context_text`: Option B serialized context with ` [TURN] ` separator.
- `context_turns_json`: Option A structured list of context turns with turn index, key, speaker, and end_time.
- `sarcasm`: binary label (`0` or `1`).
- `sarcasm_type`: category (`NONE`, `PRO`, `ILL`, `EMB`, `LIK`).
- Validation metadata: `implicit_emotion`, `explicit_emotion`, `valence`, `arousal` (validation-only).
- Modality flags: `has_text` (True), `has_audio` (False), `has_video` (False).

**Must not**
- Create or modify cross-validation folds.
- Add annotated valence or arousal as model inputs.
- Implement the shared evaluation harness or classifier head.

See [dataset](dataset.md) and [cleaning](cleaning.md).
