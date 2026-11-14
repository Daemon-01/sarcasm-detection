---
type: concept
title: Target text processing
description: Processing contract for the target utterance.
tags: [text-processing, target]
---

# Target text processing

Implemented module: `src/sarcastone/text/preprocess.py` (`prepare_target_text`, `normalize_whitespace`).

Prepare target utterance text for the text models while preserving its association with `utterance_id`.

## Text Normalization Invariants
- **Deterministic Whitespace & Newline Handling**: Normalizes `\r\n`, `\r`, `\t`, and consecutive spaces into single clean spaces; trims outer whitespace.
- **Strict Preservation**: Retains 100% of original character casing (no forced lowercasing) and punctuation (no stripping punctuation or exclamation/question marks).
- **Prohibited Transformations**: No stemming, lemmatization, stop-word removal, or unapproved text rewriting.
- **Traceability**: Every target utterance remains strictly associated with its `utterance_id` (`SCENE`).

See [cleaning](../data/cleaning.md), [context text](context-text.md) and [context ablation](../experiments/context-ablation.md).
