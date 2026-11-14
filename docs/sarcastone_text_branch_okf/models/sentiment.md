---
type: concept
title: Pre-trained sentiment probabilities
description: Three-class sentiment model outputs used by the text branch.
tags: [sentiment, probabilities, pretrained-model]
---

# Pre-trained 3-class sentiment model

Use a pre-trained 3-class sentiment model to produce sentiment probabilities.

These probabilities are a text-branch feature/output and must be associated with `utterance_id`.

Annotated MUStARD++ valence and arousal are for model validation only; they must never be supplied as input features.

## Specifications & Implementation
- **Checkpoint**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Output Classes**: 3 classes with index mapping:
  - `0`: Negative (`prob_negative`)
  - `1`: Neutral (`prob_neutral`)
  - `2`: Positive (`prob_positive`)
- **Normalization**: Softmax output probabilities summing to 1.0.
- **Precision**: float32 (`.round(4)` in tabular output).
- **Storage Locations**:
  - `features/emotion/{utterance_id}.npy`: 3-dimensional NumPy array.
  - `features/emotion/sentiment_probabilities.npz`: Compressed archive of all 1,202 vectors.
  - `features/emotion/sentiment_probabilities.csv`: Tabular CSV containing `utterance_id`, `prob_negative`, `prob_neutral`, `prob_positive`, and `predicted_sentiment`.

## Validation Against Annotated MUStARD++ Valence
- **Validation-only protocol**: Valence was compared against continuous text sentiment (`prob_positive - prob_negative`).
- **Pearson Correlation**: **+0.2182** (positive correlation confirming correct sentiment alignment).
- **Monotonic Mean Valence Across Classes**:
  - Negative: **3.6537**
  - Neutral: **4.1281**
  - Positive: **4.4052**
- **Analysis**: The moderate correlation (+0.22) reflects the linguistic nature of sarcasm in MUStARD++, where sarcastic utterances frequently employ positive literal wording ("Oh brilliant") despite negative underlying situation/valence.

See [feature export](../features/export.md) and [evaluation conventions](../evaluation/conventions.md).
