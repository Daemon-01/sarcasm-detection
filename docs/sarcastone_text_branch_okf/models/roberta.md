---
type: concept
title: RoBERTa text embeddings
description: Frozen RoBERTa-base representation for the text branch.
tags: [roberta, embeddings, frozen-encoder]
---

# RoBERTa-base embeddings

Use a pre-trained **RoBERTa-base** encoder as a frozen encoder. Produce **attention-mask-aware mean (average) pooled** text embeddings, and train a small classification head on top.

Do not train the encoder from scratch or unfreeze it unless the project plan is explicitly changed.

## Pooling Requirement: Attention-Mask-Aware Average Pooling
Padding tokens must never contribute to the pooled representation. The pooling layer must explicitly weight token hidden states by the attention mask:
$$\mathbf{e} = \frac{\sum_{i=1}^{T} \mathbf{h}_i \cdot m_i}{\sum_{i=1}^{T} m_i}$$
where $\mathbf{h}_i$ is the hidden state at token position $i$ and $m_i \in \{0, 1\}$ is the attention mask. This guarantees variable sequence lengths and padding tokens do not distort the 768-dimensional sentence embedding.

The text branch exports these 768-dimensional embeddings keyed by `utterance_id` for downstream speech/fusion integration.

## Configuration & Specifications
- **Base Checkpoint**: `roberta-base` (frozen, 12 layers, 768 hidden dimension).
- **Pooling**: Attention-mask-aware average (mean) pooling over non-padding tokens.
- **Classifier Head**: Small trainable feed-forward network (768 -> 128 -> ReLU -> Dropout(0.2) -> 2) on top of frozen embeddings.
- **Export Format**: `.npy` feature files keyed by `utterance_id` under `features/text/` and compressed archive `features/text/roberta_embeddings_utterance_only.npz`.
- **Conditions**: Supports both `utterance_only` and `utterance_plus_context` ablation inputs.

## Empirical Cross-Validation Results (5 Frozen Folds, 3 Seeds: 42, 123, 2026)
- **Utterance Only**:
  - Accuracy: **0.5980 ± 0.0044** (59.80%)
  - Macro F1: **0.5961 ± 0.0037** (59.61%)
  - Weighted F1: **0.5974 ± 0.0035** (59.74%)
- **Utterance + Context**:
  - Accuracy: **0.5887 ± 0.0035** (58.87%)
  - Macro F1: **0.5850 ± 0.0025** (58.50%)
  - Weighted F1: **0.5861 ± 0.0028** (58.61%)
- **Comparison to TF-IDF Baseline (Utterance Only)**:
  - Accuracy Gain: **+5.48%** ($p = 1.0 \times 10^{-5}$, paired t-test, $t = 6.68$, statistically significant).
  - Macro F1 Gain: **+5.63%** ($p = 1.2 \times 10^{-5}$, paired t-test, $t = 6.57$, statistically significant).

See [feature export](../features/export.md) and [context ablation](../experiments/context-ablation.md).
