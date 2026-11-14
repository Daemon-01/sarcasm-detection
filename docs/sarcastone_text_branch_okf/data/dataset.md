---
type: concept
title: Dataset
description: Dataset scope and known constraints for the text branch.
resource: MUStARD++
tags: [dataset, mustard-plus-plus]
---

# Dataset

- Use MUStARD++ only (`data/mustard++_text.csv`, 6,041 total rows).
- Total target utterances: 1,202 (50/50 balance: 601 sarcastic `1`, 601 non-sarcastic `0`).
- Total context utterances: 4,839 (across 1,202 scenes, 1–12 turns per scene, mean 4.03).
- The shared sample key is `utterance_id` (matching `SCENE` e.g. `1_10004`, `1_S09E01_027`).
- The text branch supports both an utterance-only condition (`target_text`) and an utterance-plus-context condition (`context_text`).
- Annotated valence and arousal are validation-only metadata and must never be input features.

## Verified Dataset Policies
- **Source file**: `data/mustard++_text.csv` with 12 canonical columns: `SCENE`, `KEY`, `SENTENCE`, `END_TIME`, `SPEAKER`, `SHOW`, `Sarcasm`, `Sarcasm_Type`, `Implicit_Emotion`, `Explicit_Emotion`, `Valence`, `Arousal`.
- **Target vs. Context**: Utterances with non-null `Sarcasm` are target utterances (always the final turn of each scene). Utterances with null `Sarcasm` are context utterances.
- **Corrupted Key Resolution**: In scene `1_S11E03_067` (row 2596), the target `KEY` was accidentally annotated as `'Disgust'`. Normalized to `1_S11E03_067_u` while preserving `raw_target_key = 'Disgust'`.
- **Corrupted Show Resolution**: In scene `1_2837` (row 613), context `SHOW` was annotated as `'0'`. Canonical show inferred as `'BBT'`.
- **Gap Preservation**: Turn numbers in context are sorted numerically by extracting the integer suffix; 4 scenes contain gaps (`[0, 1, 4, 5]`, `[0, 1, 3]`, `[0..5, 7]`, `[0..10, 12]`) which are preserved without fabrication.

See [loader](loader.md), [cleaning](cleaning.md) and [context representation](context.md).
