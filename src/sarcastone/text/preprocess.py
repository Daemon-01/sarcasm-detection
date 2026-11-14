"""Text preprocessing and context serialization module for SarcasTone.

Implements text normalization and context preparation in accordance with
AGENTS.md Section 4 & 5 and OKF specifications:
- Preserves original wording, capitalization, and punctuation.
- Deterministically normalizes whitespace and line breaks.
- Strictly forbids destructive transformations (no lowercasing, stemming,
  lemmatization, or stopword removal).
- Maintains Option A (structured context turns) and Option B (serialized with [TURN]).
- Supports both experimental conditions:
  - "utterance_only": model input contains target utterance text alone.
  - "utterance_plus_context": model input combines conversational context and target.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Standard turn separator agreed in AGENTS.md Section 5
TURN_DELIMITER = " [TURN] "

# Context ablation conditions as defined in OKF experiments/context-ablation.md
UTTERANCE_ONLY = "utterance_only"
UTTERANCE_PLUS_CONTEXT = "utterance_plus_context"
VALID_CONDITIONS = {UTTERANCE_ONLY, UTTERANCE_PLUS_CONTEXT}


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace and line breaks deterministically.

    Preserves exact casing and punctuation while converting internal CRLF / LF
    and consecutive whitespace into single spaces, stripping outer boundaries.

    Args:
        text: Input string.

    Returns:
        Whitespace-normalized string.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    # Replace carriage returns and newlines with space
    cleaned = re.sub(r"[\r\n\t]+", " ", text)
    # Collapse multiple consecutive horizontal spaces to a single space
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def prepare_target_text(text: str) -> str:
    """Prepare target utterance text.

    Args:
        text: Raw target utterance text.

    Returns:
        Cleaned target text with casing and punctuation preserved.
    """
    return normalize_whitespace(text)


def prepare_context_turn(
    text: str,
    speaker: Optional[str] = None,
    include_speaker: bool = False,
) -> str:
    """Prepare a single context turn text.

    Args:
        text: Raw turn sentence text.
        speaker: Optional character name.
        include_speaker: Whether to prepend the speaker name (e.g. 'SHELDON: ...').

    Returns:
        Prepared context turn string.
    """
    norm_text = normalize_whitespace(text)
    if include_speaker and speaker and speaker.strip() not in {"", "-"}:
        return f"{speaker.strip()}: {norm_text}"
    return norm_text


def serialize_context(
    turns: List[Dict[str, Any]],
    delimiter: str = TURN_DELIMITER,
    include_speaker: bool = False,
) -> str:
    """Convert Option A structured turns into Option B serialized string.

    Preserves chronological order and sequence gaps.

    Args:
        turns: List of turn dicts containing at least 'text' and optionally 'speaker'.
        delimiter: Separator token between turns (default: ' [TURN] ').
        include_speaker: Whether to prefix turns with speaker names.

    Returns:
        Serialized context string.
    """
    if not turns:
        return ""
    formatted_turns = [
        prepare_context_turn(t.get("text", ""), speaker=t.get("speaker"), include_speaker=include_speaker)
        for t in turns
    ]
    return delimiter.join(formatted_turns)


def deserialize_context(context_text: str, delimiter: str = TURN_DELIMITER) -> List[str]:
    """Split serialized context string back into individual turn strings.

    Args:
        context_text: Serialized context text.
        delimiter: Separator token used for serialization.

    Returns:
        List of turn text strings.
    """
    if not context_text or not context_text.strip():
        return []
    return [t.strip() for t in context_text.split(delimiter) if t.strip()]


@dataclass
class TextSample:
    """Represents a preprocessed multimodal sarcasm text sample."""

    utterance_id: str
    target_text: str
    context_text: str = ""
    context_turns: List[Dict[str, Any]] = field(default_factory=list)
    target_speaker: str = ""
    show: str = ""
    label: Optional[int] = None
    sarcasm_type: Optional[str] = None
    valence: Optional[float] = None
    arousal: Optional[float] = None
    condition: str = UTTERANCE_ONLY

    def get_input_text(
        self,
        condition: Optional[str] = None,
        context_target_separator: str = " </s> ",
    ) -> str:
        """Get model-ready text based on requested experimental condition.

        Args:
            condition: 'utterance_only' or 'utterance_plus_context'. Defaults to self.condition.
            context_target_separator: Separator token between context and target when combining.

        Returns:
            Text string for tokenizer or vectorizer.
        """
        cond = condition if condition is not None else self.condition
        if cond not in VALID_CONDITIONS:
            raise ValueError(f"Unknown condition '{cond}'. Expected one of {sorted(VALID_CONDITIONS)}")

        if cond == UTTERANCE_ONLY:
            return self.target_text

        # Utterance plus context condition
        if self.context_text:
            return f"{self.context_text}{context_target_separator}{self.target_text}"
        return self.target_text


def process_sample(
    record: Union[pd.Series, Dict[str, Any]],
    condition: str = UTTERANCE_ONLY,
) -> TextSample:
    """Convert a manifest row or dictionary into a structured TextSample.

    Args:
        record: Series or Dict with manifest columns.
        condition: 'utterance_only' or 'utterance_plus_context'.

    Returns:
        Validated TextSample instance.
    """
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"Invalid condition '{condition}'. Expected one of {sorted(VALID_CONDITIONS)}")

    def get_val(key: str, default: Any = None) -> Any:
        val = record.get(key, default) if isinstance(record, dict) else record[key] if key in record else default
        return val if pd.notnull(val) else default

    utterance_id = str(get_val("utterance_id", get_val("scene_id", ""))).strip()
    target_text = prepare_target_text(str(get_val("target_text", "")))

    raw_context_turns = get_val("context_turns_json", None)
    if raw_context_turns:
        if isinstance(raw_context_turns, str):
            try:
                context_turns = json.loads(raw_context_turns)
            except json.JSONDecodeError:
                logger.warning("Could not parse context_turns_json for %s", utterance_id)
                context_turns = []
        elif isinstance(raw_context_turns, list):
            context_turns = raw_context_turns
        else:
            context_turns = []
    else:
        context_turns = []

    raw_context_text = get_val("context_text", "")
    if raw_context_text:
        context_text = normalize_whitespace(str(raw_context_text))
    elif context_turns:
        context_text = serialize_context(context_turns)
    else:
        context_text = ""

    label_raw = get_val("sarcasm", None)
    label = int(label_raw) if label_raw is not None else None

    return TextSample(
        utterance_id=utterance_id,
        target_text=target_text,
        context_text=context_text,
        context_turns=context_turns,
        target_speaker=str(get_val("target_speaker", "")).strip(),
        show=str(get_val("show", "")).strip(),
        label=label,
        sarcasm_type=str(get_val("sarcasm_type", "")).strip() or None,
        valence=float(get_val("valence")) if get_val("valence") is not None else None,
        arousal=float(get_val("arousal")) if get_val("arousal") is not None else None,
        condition=condition,
    )


def process_manifest(
    manifest_df: pd.DataFrame,
    condition: str = UTTERANCE_ONLY,
) -> List[TextSample]:
    """Process an entire manifest DataFrame into a list of TextSamples.

    Args:
        manifest_df: Master manifest DataFrame (e.g. loaded from data/manifest.csv).
        condition: 'utterance_only' or 'utterance_plus_context'.

    Returns:
        List of TextSample objects.
    """
    logger.info("Preprocessing %d samples under '%s' condition", len(manifest_df), condition)
    samples: List[TextSample] = []
    for _, row in manifest_df.iterrows():
        samples.append(process_sample(row, condition=condition))
    return samples


def get_text_inputs_and_labels(
    manifest_df: pd.DataFrame,
    condition: str = UTTERANCE_ONLY,
    context_target_separator: str = " </s> ",
) -> Tuple[List[str], np.ndarray, List[str]]:
    """Extract model-ready text inputs, label array, and utterance IDs from manifest.

    Args:
        manifest_df: Master manifest DataFrame.
        condition: 'utterance_only' or 'utterance_plus_context'.
        context_target_separator: Separator token between context and target when combining.

    Returns:
        Tuple of (texts, labels_array, utterance_ids).
    """
    samples = process_manifest(manifest_df, condition=condition)
    texts = [s.get_input_text(context_target_separator=context_target_separator) for s in samples]
    labels = np.array([s.label for s in samples if s.label is not None], dtype=np.int64)
    utterance_ids = [s.utterance_id for s in samples]
    return texts, labels, utterance_ids

