---
type: concept
title: Context text processing
description: Processing contract for associated context text.
tags: [text-processing, context]
---

# Context text processing

Implemented module: `src/sarcastone/text/preprocess.py` (`serialize_context`, `deserialize_context`, `prepare_context_turn`).

Prepare the context associated with each target `utterance_id` for the utterance-plus-context condition.

## Verified Context Processing Specification
- **Turn Ordering**: Context turns are sorted strictly in ascending numeric order extracted from `_c_(\d+)$`.
- **Gaps**: Sequence gaps are preserved without fabricating missing turns or renumbering.
- **Option B Serialization**: Context turns are joined using the agreed delimiter ` [TURN] `.
- **Option A Structured Recovery**: Context turns can be individually recovered via `deserialize_context` or `context_turns_json`.
- **Speaker Representation**: Supported with optional speaker prefixing (`SPEAKER: text`).
- **Condition Separation**: Context text is never merged into the `utterance_only` condition.

See [context representation](../data/context.md) and [context ablation](../experiments/context-ablation.md).

See [context representation](../data/context.md) and [context ablation](../experiments/context-ablation.md).
