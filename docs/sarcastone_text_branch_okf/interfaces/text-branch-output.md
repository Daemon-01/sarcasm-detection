---
type: interface
title: Text branch output interface
description: Single handoff stub for the speech/fusion team.
tags: [interface, handoff]
---

# Text branch → speech/fusion handoff

The text branch provides, per `utterance_id`:

| Output | Description | Format & Path |
|---|---|---|
| Text embeddings | Attention-mask-aware average (mean) pooled embeddings (768-d) from frozen RoBERTa-base | `features/text/{utterance_id}.npy` (shape `(768,)`, float32) or `features/text/roberta_embeddings_utterance_only.npz` |
| Sentiment probabilities | Probabilities from the pre-trained 3-class sentiment model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) | `features/emotion/{utterance_id}.npy` (shape `(3,)`, float32 `[neg, neu, pos]`) or `features/emotion/sentiment_probabilities.csv` |

**Shared convention:** all features are keyed strictly by `utterance_id` matching `data/manifest.csv`.

**Handoff Specifications Confirmed:**
- File format: Individual `.npy` binaries and consolidated `.npz` archives.
- Embedding dimensions: 768 float32 (RoBERTa), 3 float32 (sentiment).
- Sentiment class ordering: `[0: negative, 1: neutral, 2: positive]`.
- Input conditions: Primary text features represent target utterance (`utterance_only`), with serialized context variants (`utterance_plus_context`) available in `roberta_embeddings_utterance_plus_context.npz`.
- Missing records: 0 missing records (all 1,202 target utterances are present and verified).

This file defines the text branch's handoff only. Speech processing and multimodal fusion are outside this bundle.

See [feature export](../features/export.md).
