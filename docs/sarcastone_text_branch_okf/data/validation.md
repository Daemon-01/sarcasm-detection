---
type: concept
title: Data validation
description: Checks required before text features are generated.
tags: [validation, data-quality]
---

# Data validation

Implemented module: `validate_manifest` inside `src/sarcastone/data/manifest.py`.

## Validation Invariants Enforced
- Exactly 1,202 total target samples in `manifest.csv`.
- Exactly 6,041 total turns in `manifest_utterances.csv`.
- Perfect binary balance: exactly 601 non-sarcastic (`sarcasm=0`, `sarcasm_type='NONE'`) and 601 sarcastic (`sarcasm=1`, `sarcasm_type` in `{'PRO', 'ILL', 'EMB', 'LIK'}`).
- 100% unique `utterance_id` and `scene_id` values (zero duplicate samples).
- 0 empty or whitespace-only target text entries.
- Context turns strictly sorted by integer turn suffix with turn gaps preserved.
- Valence and arousal scores verified within valid discrete rating bounds [1.0, 9.0].
- Modality flags correctly tracked: `has_text=True`, `has_audio=False`, `has_video=False`.

Validation failures raise explicit `ValueError` exceptions rather than silently dropping or modifying data.

See [loader](loader.md), [cleaning](cleaning.md) and [dataset](dataset.md).
