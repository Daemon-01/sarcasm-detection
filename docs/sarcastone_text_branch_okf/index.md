---
type: index
title: SarcasTone text branch
description: Navigation for the text-only branch knowledge bundle.
tags: [sarcastone, text, okf]
---

# SarcasTone — text branch

This bundle covers the text branch only. It is a Markdown-only implementation plan, not executable Python code.

## Project
- [Project overview](project/overview.md)
- [Dataset](data/dataset.md)

## Data pipeline
- [Loader](data/loader.md)
- [Cleaning and sample construction](data/cleaning.md)
- [Context representation](data/context.md)
- [Validation](data/validation.md)
- [Shared folds](data/folds.md)

## Text processing and models
- [Target text processing](text/target-text.md)
- [Context text processing](text/context-text.md)
- [TF-IDF baseline](models/tfidf-logreg.md)
- [RoBERTa embeddings](models/roberta.md)
- [Sentiment probabilities](models/sentiment.md)

## Experiments and outputs
- [Context ablation](experiments/context-ablation.md)
- [Feature export](features/export.md)
- [Evaluation conventions](evaluation/conventions.md)
- [Speech/fusion interface stub](interfaces/text-branch-output.md)

## Change history
- [Log](log.md)
