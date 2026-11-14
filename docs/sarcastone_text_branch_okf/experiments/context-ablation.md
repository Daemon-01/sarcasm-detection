---
type: concept
title: Utterance and context ablation
description: Controlled comparison of target-only and context-aware text inputs.
tags: [ablation, context, experiment]
---

# Context ablation

Compare these text input conditions:

1. **Utterance only:** target utterance text (`target_text`).
2. **Utterance + context:** target utterance combined with serialized context text (`context_text </s> target_text`).

Evaluation uses the shared frozen folds (`data/folds.json`) and 3 seeds (42, 123, 2026).

## Context Ablation Experimental Results

### Baseline (TF-IDF + Logistic Regression)
- **Utterance Only**: Accuracy = 54.32% ± 0.00%, Macro F1 = 53.98% ± 0.00%.
- **Utterance + Context**: Accuracy = 56.59% ± 0.00%, Macro F1 = 56.23% ± 0.00%.
- **Context Gain**: **+2.27% accuracy, +2.25% Macro F1**. In n-gram models, conversational turns provide additional vocabulary tokens indicating topical tension.

### RoBERTa-base (Frozen + Attention-Mask-Aware Mean Pooling + Classification Head)
- **Utterance Only**: Accuracy = **59.80% ± 0.44%**, Macro F1 = **59.61% ± 0.37%**.
- **Utterance + Context**: Accuracy = **58.87% ± 0.35%**, Macro F1 = **58.50% ± 0.25%**.
- **Analysis**: Mean-pooling across long concatenated dialogues (up to 12 turns) dilutes the target utterance's concise sarcastic cues with neutral dialogue tokens in a frozen encoder. Consequently, target utterance only yields superior dense semantic representation (+0.93% accuracy) for frozen RoBERTa-base sentence embedding.

See [context representation](../data/context.md), [RoBERTa](../models/roberta.md) and [evaluation conventions](../evaluation/conventions.md).
