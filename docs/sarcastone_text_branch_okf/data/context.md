---
type: concept
title: Context representation
description: How context is retained for the utterance-context ablation.
tags: [context, representation, ablation]
---

# Context representation

The text branch supports two conditions:

1. **Utterance only:** model input contains `target_text`.
2. **Utterance plus context:** model input contains `target_text` and associated `context_text`.

Context association is traceable directly via `utterance_id` (`SCENE`).

## Verified Representations in Master Manifest
- **Option A (Structured):** Stored in `context_turns_json` as an ordered JSON list of turn objects:
  `[{"turn": int, "key": str, "speaker": str, "text": str, "end_time": str}, ...]`.
  Individual context turns remain fully recoverable with metadata.
- **Option B (Serialized):** Stored in `context_text` using the agreed turn delimiter:
  `turn_0 [TURN] turn_1 [TURN] turn_2 ...`
- **Turn Ordering & Gaps:** Suffixes are parsed as integers (`_c_(\d+)$`). Gaps in turn numbers are preserved without renumbering or fabricating missing turns.

See [cleaning](cleaning.md), [context text processing](../text/context-text.md) and [context ablation](../experiments/context-ablation.md).
