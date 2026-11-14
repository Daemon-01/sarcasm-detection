---
type: concept
title: Text feature export
description: Export contract for downstream speech/fusion integration.
tags: [features, export, interoperability]
---

# Text feature export

Exports all text-branch features required by the downstream speech/fusion team (Team B):

1. **Text embeddings**:
   - Source: Frozen `roberta-base` with attention-mask-aware average (mean) pooling.
   - Tensor Shape: `(768,)` float32.
   - File locations:
     - Individual files: `features/text/{utterance_id}.npy` (1,202 files).
     - Condition-specific files: `features/text/{utterance_id}_utterance_only.npy` and `features/text/{utterance_id}_utterance_plus_context.npy`.
     - Compressed archive: `features/text/roberta_embeddings_utterance_only.npz` and `features/text/roberta_embeddings_utterance_plus_context.npz`.

2. **Three-class sentiment probabilities**:
   - Source: Pre-trained `cardiffnlp/twitter-roberta-base-sentiment-latest`.
   - Tensor Shape: `(3,)` float32 representing `[prob_negative, prob_neutral, prob_positive]`.
   - File locations:
     - Individual files: `features/emotion/{utterance_id}.npy` (1,202 files).
     - Compressed archive: `features/emotion/sentiment_probabilities.npz`.
     - Tabular CSV: `features/emotion/sentiment_probabilities.csv`.

Every exported record is strictly keyed by `utterance_id` matching `data/manifest.csv`.

See [text branch output interface](../interfaces/text-branch-output.md), [RoBERTa](../models/roberta.md) and [sentiment model](../models/sentiment.md).
