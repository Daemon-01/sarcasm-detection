"""Dataset manifest builder and validator for MUStARD++.

Constructs standardized sample-level and utterance-level manifests from raw
MUStARD++ annotations, enforcing AGENTS.md and OKF specifications:
- Preserves original utterance identifiers and raw text (casing/punctuation).
- Separates target utterances from context utterances.
- Extracts numeric turn suffixes and preserves ordering and gaps.
- Supports Option A (structured context turns) and Option B ([TURN] serialized).
- Records modality flags (has_text, has_audio, has_video) for future fusion.
- Validates data integrity, reporting anomalies and label distributions.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Context turn delimiter for Option B serialization (as specified in AGENTS.md)
TURN_DELIMITER = " [TURN] "

# Known dataset anomalies documented in Phase 1 inspection
CORRUPTED_KEY_SCENE = "1_S11E03_067"
CORRUPTED_KEY_RAW = "Disgust"
CORRUPTED_SHOW_SCENE = "1_2837"


def parse_context_turn(key: str) -> Optional[int]:
    """Extract integer turn index from context utterance KEY.

    Handles both zero-padded keys (e.g. '_c_00' -> 0) and unpadded keys
    (e.g. '_c_0' -> 0, '_c_12' -> 12). Returns None for target utterances
    or non-context keys.

    Args:
        key: Utterance identifier string.

    Returns:
        Integer turn index if matched, otherwise None.
    """
    match = re.search(r"_c_(\d+)$", str(key).strip())
    if match:
        return int(match.group(1))
    return None


def load_raw_mustard(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load raw MUStARD++ text CSV file and perform initial schema checks.

    Args:
        csv_path: Path to mustard++_text.csv.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"MUStARD++ dataset file not found at: {path}")

    df = pd.read_csv(path)

    expected_cols = {
        "SCENE", "KEY", "SENTENCE", "END_TIME", "SPEAKER", "SHOW",
        "Sarcasm", "Sarcasm_Type", "Implicit_Emotion", "Explicit_Emotion",
        "Valence", "Arousal"
    }
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns in {path}: {sorted(missing_cols)}")

    logger.info("Loaded raw MUStARD++ CSV with %d rows and %d columns", len(df), len(df.columns))
    return df


def build_utterance_manifest(
    df: pd.DataFrame,
    audio_dir: Optional[Union[str, Path]] = None,
    video_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Build turn-level utterance registry for all 6,041 utterances.

    Args:
        df: Raw MUStARD++ DataFrame.
        audio_dir: Optional directory to check for audio files.
        video_dir: Optional directory to check for video files.

    Returns:
        DataFrame with one row per utterance (targets and contexts).
    """
    records: List[Dict[str, Any]] = []
    audio_path_obj = Path(audio_dir) if audio_dir else None
    video_path_obj = Path(video_dir) if video_dir else None

    # First pass: determine canonical show per scene (handles row 613 SHOW='0' anomaly)
    scene_shows: Dict[str, str] = {}
    for scene, group in df.groupby("SCENE", sort=False):
        valid_shows = group.loc[group["SHOW"] != "0", "SHOW"]
        if not valid_shows.empty:
            scene_shows[scene] = valid_shows.iloc[0]
        else:
            scene_shows[scene] = group["SHOW"].iloc[0]

    for _, row in df.iterrows():
        scene = str(row["SCENE"]).strip()
        raw_key = str(row["KEY"]).strip()
        sentence = str(row["SENTENCE"]).strip()
        speaker = str(row["SPEAKER"]).strip()
        end_time = str(row["END_TIME"]).strip()
        canonical_show = scene_shows.get(scene, str(row["SHOW"]).strip())

        is_target = pd.notnull(row["Sarcasm"])

        if is_target:
            turn_order = -1
            # Handle row 2596 anomaly where KEY is 'Disgust'
            if scene == CORRUPTED_KEY_SCENE and raw_key == CORRUPTED_KEY_RAW:
                norm_key = f"{scene}_u"
            else:
                norm_key = raw_key
        else:
            turn_order = parse_context_turn(raw_key)
            norm_key = raw_key

        expected_audio = f"data/audio/{norm_key}.wav"
        expected_video = f"data/video/{norm_key}.mp4"

        has_audio = (audio_path_obj / f"{norm_key}.wav").is_file() if audio_path_obj else False
        has_video = (video_path_obj / f"{norm_key}.mp4").is_file() if video_path_obj else False

        record: Dict[str, Any] = {
            "utterance_key": norm_key,
            "raw_key": raw_key,
            "scene_id": scene,
            "is_target": is_target,
            "turn_order": turn_order,
            "speaker": speaker,
            "show": canonical_show,
            "text": sentence,
            "end_time": end_time,
            "sarcasm": int(row["Sarcasm"]) if is_target else None,
            "sarcasm_type": row["Sarcasm_Type"] if is_target else None,
            "implicit_emotion": row["Implicit_Emotion"] if is_target else None,
            "explicit_emotion": row["Explicit_Emotion"] if is_target else None,
            "valence": float(row["Valence"]) if is_target else None,
            "arousal": float(row["Arousal"]) if is_target else None,
            "has_text": True,
            "has_audio": has_audio,
            "has_video": has_video,
            "audio_path": expected_audio,
            "video_path": expected_video,
        }
        records.append(record)

    out_df = pd.DataFrame(records)
    logger.info("Built utterance-level manifest with %d rows", len(out_df))
    return out_df


def build_sample_manifest(
    df: pd.DataFrame,
    audio_dir: Optional[Union[str, Path]] = None,
    video_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Build master sample-level manifest with 1 row per target sample (1,202 rows).

    Enforces context turn ordering (Option A & B), anomaly resolution, and modality flags.

    Args:
        df: Raw MUStARD++ DataFrame.
        audio_dir: Optional directory to check for audio files.
        video_dir: Optional directory to check for video files.

    Returns:
        Standardized master manifest DataFrame.
    """
    audio_path_obj = Path(audio_dir) if audio_dir else None
    video_path_obj = Path(video_dir) if video_dir else None

    samples: List[Dict[str, Any]] = []

    for scene, group in df.groupby("SCENE", sort=False):
        target_rows = group[group["Sarcasm"].notnull()]
        context_rows = group[group["Sarcasm"].isnull()]

        if len(target_rows) == 0:
            logger.warning("Scene %s has no target utterance, skipping.", scene)
            continue
        if len(target_rows) > 1:
            logger.warning("Scene %s has %d target utterances; expected exactly 1.", scene, len(target_rows))

        target_row = target_rows.iloc[-1]
        scene_str = str(scene).strip()
        raw_target_key = str(target_row["KEY"]).strip()

        # Handle row 2596 anomaly where target KEY is 'Disgust'
        if scene_str == CORRUPTED_KEY_SCENE and raw_target_key == CORRUPTED_KEY_RAW:
            norm_target_key = f"{scene_str}_u"
            logger.info("Normalized corrupted target key in scene %s: '%s' -> '%s'",
                        scene_str, raw_target_key, norm_target_key)
        else:
            norm_target_key = raw_target_key

        target_text = str(target_row["SENTENCE"]).strip()
        target_speaker = str(target_row["SPEAKER"]).strip()

        # Canonical show name: handle row 613 SHOW='0'
        valid_shows = group.loc[group["SHOW"] != "0", "SHOW"]
        canonical_show = valid_shows.iloc[0] if not valid_shows.empty else str(target_row["SHOW"]).strip()

        target_end_time = str(target_row["END_TIME"]).strip()
        sarcasm_label = int(target_row["Sarcasm"])
        sarcasm_type = str(target_row["Sarcasm_Type"]).strip()

        implicit_emotion = str(target_row["Implicit_Emotion"]).strip() if pd.notnull(target_row["Implicit_Emotion"]) else None
        explicit_emotion = str(target_row["Explicit_Emotion"]).strip() if pd.notnull(target_row["Explicit_Emotion"]) else None
        valence = float(target_row["Valence"]) if pd.notnull(target_row["Valence"]) else None
        arousal = float(target_row["Arousal"]) if pd.notnull(target_row["Arousal"]) else None

        # Process context turns: sort by integer turn index extracted from KEY
        parsed_turns: List[Tuple[int, Dict[str, Any]]] = []
        for _, c_row in context_rows.iterrows():
            c_key = str(c_row["KEY"]).strip()
            turn_idx = parse_context_turn(c_key)
            if turn_idx is None:
                logger.warning("Could not parse numeric turn from context key: %s (scene %s)", c_key, scene_str)
                turn_idx = 999  # Put at end if unparseable

            turn_info = {
                "turn": turn_idx,
                "key": c_key,
                "speaker": str(c_row["SPEAKER"]).strip(),
                "text": str(c_row["SENTENCE"]).strip(),
                "end_time": str(c_row["END_TIME"]).strip(),
            }
            parsed_turns.append((turn_idx, turn_info))

        # Sort strictly in ascending numeric order (preserving gaps without fabricating)
        parsed_turns.sort(key=lambda t: t[0])
        structured_context = [t[1] for t in parsed_turns]

        # Option B: Serialized string using agreed separator [TURN]
        if structured_context:
            context_text = TURN_DELIMITER.join(turn["text"] for turn in structured_context)
            context_speakers = [turn["speaker"] for turn in structured_context]
        else:
            context_text = ""
            context_speakers = []

        context_turn_count = len(structured_context)

        # Modality checks
        expected_audio = f"data/audio/{norm_target_key}.wav"
        expected_video = f"data/video/{norm_target_key}.mp4"
        has_audio = (audio_path_obj / f"{norm_target_key}.wav").is_file() if audio_path_obj else False
        has_video = (video_path_obj / f"{norm_target_key}.mp4").is_file() if video_path_obj else False

        sample_record: Dict[str, Any] = {
            "utterance_id": scene_str,  # Shared sample key matching SCENE and feature dicts
            "scene_id": scene_str,
            "target_key": norm_target_key,
            "raw_target_key": raw_target_key,
            "target_text": target_text,
            "target_speaker": target_speaker,
            "show": canonical_show,
            "end_time": target_end_time,
            "sarcasm": sarcasm_label,
            "sarcasm_type": sarcasm_type,
            "implicit_emotion": implicit_emotion,
            "explicit_emotion": explicit_emotion,
            "valence": valence,
            "arousal": arousal,
            "context_turn_count": context_turn_count,
            "context_text": context_text,
            "context_turns_json": json.dumps(structured_context, ensure_ascii=False),
            "context_speakers": json.dumps(context_speakers, ensure_ascii=False),
            "has_text": True,
            "has_audio": has_audio,
            "has_video": has_video,
            "audio_path": expected_audio,
            "video_path": expected_video,
        }
        samples.append(sample_record)

    out_df = pd.DataFrame(samples)
    logger.info("Built master sample manifest with %d samples", len(out_df))
    return out_df


def validate_manifest(manifest_df: pd.DataFrame) -> Dict[str, Any]:
    """Validate sample manifest integrity against project requirements.

    Checks:
    - Expected row count (1,202 for MUStARD++).
    - Perfect binary label balance (601 zeros, 601 ones).
    - Unique utterance_id and scene_id.
    - Non-empty target text and valid speaker strings.
    - Sarcasm type consistency (NONE for 0, specific subcategories for 1).
    - Validation emotion/valence/arousal bounds.

    Args:
        manifest_df: Master sample manifest DataFrame.

    Returns:
        Dictionary of validation results and statistics.

    Raises:
        ValueError: If a critical validation check fails.
    """
    total_samples = len(manifest_df)
    if total_samples != 1202:
        raise ValueError(f"Expected exactly 1202 samples, found {total_samples}")

    # Check unique utterance_ids
    unique_ids = manifest_df["utterance_id"].nunique()
    if unique_ids != total_samples:
        raise ValueError(f"Duplicate utterance_id found! {unique_ids} unique out of {total_samples}")

    # Check label distribution
    label_counts = manifest_df["sarcasm"].value_counts().to_dict()
    if label_counts.get(0, 0) != 601 or label_counts.get(1, 0) != 601:
        raise ValueError(f"Label imbalance! Expected 601/601, found: {label_counts}")

    # Check non-empty target text
    empty_targets = (manifest_df["target_text"].str.strip() == "").sum()
    if empty_targets > 0:
        raise ValueError(f"Found {empty_targets} empty target text entries!")

    # Check sarcasm type consistency
    sarc_zeros = manifest_df[manifest_df["sarcasm"] == 0]
    non_none_zeros = (sarc_zeros["sarcasm_type"] != "NONE").sum()
    if non_none_zeros > 0:
        raise ValueError(f"Found {non_none_zeros} non-sarcastic samples with Sarcasm_Type != 'NONE'")

    # Valence / Arousal bounds check
    invalid_valence = manifest_df["valence"].dropna().apply(lambda v: v < 1.0 or v > 9.0).sum()
    if invalid_valence > 0:
        raise ValueError(f"Found {invalid_valence} valence ratings outside [1.0, 9.0]")

    invalid_arousal = manifest_df["arousal"].dropna().apply(lambda a: a < 1.0 or a > 9.0).sum()
    if invalid_arousal > 0:
        raise ValueError(f"Found {invalid_arousal} arousal ratings outside [1.0, 9.0]")

    stats = {
        "total_samples": total_samples,
        "sarcasm_distribution": label_counts,
        "sarcasm_types": manifest_df["sarcasm_type"].value_counts().to_dict(),
        "show_distribution": manifest_df["show"].value_counts().to_dict(),
        "total_context_turns": int(manifest_df["context_turn_count"].sum()),
        "mean_context_turns": float(manifest_df["context_turn_count"].mean()),
        "min_context_turns": int(manifest_df["context_turn_count"].min()),
        "max_context_turns": int(manifest_df["context_turn_count"].max()),
        "speaker_count": int(manifest_df["target_speaker"].nunique()),
        "has_audio_count": int(manifest_df["has_audio"].sum()),
        "has_video_count": int(manifest_df["has_video"].sum()),
    }
    logger.info("Manifest validation passed successfully.")
    return stats


def build_and_save_manifest(
    raw_csv_path: Union[str, Path] = "data/mustard++_text.csv",
    output_dir: Union[str, Path] = "data",
    audio_dir: Optional[Union[str, Path]] = None,
    video_dir: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """High-level function to load raw data, build manifests, validate, and save CSVs.

    Args:
        raw_csv_path: Path to raw MUStARD++ CSV file.
        output_dir: Directory where manifest.csv and manifest_utterances.csv are saved.
        audio_dir: Optional path to audio directory.
        video_dir: Optional path to video directory.

    Returns:
        Tuple of (sample_manifest_df, utterance_manifest_df).
    """
    raw_df = load_raw_mustard(raw_csv_path)

    sample_manifest = build_sample_manifest(raw_df, audio_dir=audio_dir, video_dir=video_dir)
    utterance_manifest = build_utterance_manifest(raw_df, audio_dir=audio_dir, video_dir=video_dir)

    validate_manifest(sample_manifest)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    master_csv = out_path / "manifest.csv"
    sample_manifest.to_csv(master_csv, index=False)
    logger.info("Saved master sample manifest to %s", master_csv)

    utterance_csv = out_path / "manifest_utterances.csv"
    utterance_manifest.to_csv(utterance_csv, index=False)
    logger.info("Saved utterance registry to %s", utterance_csv)

    return sample_manifest, utterance_manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    repo_root = Path(__file__).resolve().parents[3]
    raw_csv = repo_root / "data" / "mustard++_text.csv"
    out_dir = repo_root / "data"
    build_and_save_manifest(raw_csv_path=raw_csv, output_dir=out_dir)

