"""Speaker-independent cross-validation fold generation, loading, and validation.

Implements the experimental protocol specified in AGENTS.md Section 7 and
OKF docs/sarcastone_text_branch_okf/data/folds.md:
- Speaker-independent 5-fold cross-validation.
- Groups samples strictly by speaker to guarantee zero speaker leakage.
- Generates and freezes fold assignments to data/folds.json.
- Reusable across baseline (TF-IDF + LogReg), RoBERTa, and multimodal fusion.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

logger = logging.getLogger(__name__)

DEFAULT_FOLDS_PATH = "data/folds.json"
DEFAULT_MANIFEST_PATH = "data/manifest.csv"
N_SPLITS = 5


def generate_speaker_folds(
    manifest_df: pd.DataFrame,
    n_splits: int = N_SPLITS,
    group_col: str = "target_speaker",
) -> Dict[str, Any]:
    """Generate speaker-independent cross-validation fold assignments.

    Uses GroupKFold on the speaker column to partition samples such that no
    speaker appears in both the train and test sets of any fold.

    Args:
        manifest_df: Master sample manifest DataFrame (1,202 samples).
        n_splits: Number of cross-validation folds (default: 5).
        group_col: Column name identifying the speaker (default: 'target_speaker').

    Returns:
        Dictionary containing fold assignments, metadata, and per-fold statistics.
    """
    if group_col not in manifest_df.columns:
        raise ValueError(f"Column '{group_col}' not found in manifest DataFrame.")

    total_samples = len(manifest_df)
    utterance_ids = manifest_df["utterance_id"].astype(str).tolist()
    speakers = manifest_df[group_col].astype(str).values
    labels = manifest_df["sarcasm"].astype(int).values

    gkf = GroupKFold(n_splits=n_splits)
    X = np.arange(total_samples)

    folds_list: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, labels, groups=speakers)):
        train_uids = [utterance_ids[i] for i in train_idx]
        test_uids = [utterance_ids[i] for i in test_idx]

        train_speakers = sorted(list(set(speakers[train_idx])))
        test_speakers = sorted(list(set(speakers[test_idx])))

        # Verify zero leakage during generation
        overlap = set(train_speakers).intersection(set(test_speakers))
        if overlap:
            raise RuntimeError(f"Speaker leakage detected in fold {fold_idx}: {overlap}")

        test_labels = labels[test_idx]
        sarc_count = int((test_labels == 1).sum())
        nonsarc_count = int((test_labels == 0).sum())
        sarc_ratio = float(sarc_count / len(test_idx)) if len(test_idx) > 0 else 0.0

        fold_record: Dict[str, Any] = {
            "fold": fold_idx,
            "train_sample_count": len(train_idx),
            "test_sample_count": len(test_idx),
            "test_sarcastic_count": sarc_count,
            "test_nonsarcastic_count": nonsarc_count,
            "test_sarcasm_ratio": round(sarc_ratio, 4),
            "train_speakers": train_speakers,
            "test_speakers": test_speakers,
            "train_utterance_ids": train_uids,
            "test_utterance_ids": test_uids,
        }
        folds_list.append(fold_record)

    folds_data: Dict[str, Any] = {
        "metadata": {
            "protocol": "speaker_independent_cross_validation",
            "n_splits": n_splits,
            "total_samples": total_samples,
            "group_column": group_col,
            "unique_speakers": int(len(set(speakers))),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generator": "sklearn.model_selection.GroupKFold",
        },
        "folds": folds_list,
    }

    logger.info("Generated %d speaker-independent folds across %d samples.", n_splits, total_samples)
    return folds_data


def validate_folds_against_manifest(
    folds_data: Dict[str, Any],
    manifest_df: pd.DataFrame,
    group_col: str = "target_speaker",
) -> Dict[str, Any]:
    """Validate frozen fold assignments against manifest data to guarantee integrity.

    Checks:
    - Number of folds equals expected n_splits.
    - Test sets strictly partition the manifest (disjoint union equals all sample IDs).
    - ZERO speaker leakage: no speaker appears in both train and test in any fold.
    - Class presence: both labels (0 and 1) exist in train and test sets for all folds.

    Args:
        folds_data: Fold assignments dictionary.
        manifest_df: Master sample manifest DataFrame.
        group_col: Speaker column name.

    Returns:
        Summary statistics dictionary.

    Raises:
        ValueError: If any validation check fails.
    """
    metadata = folds_data.get("metadata", {})
    n_splits = metadata.get("n_splits", len(folds_data.get("folds", [])))
    folds = folds_data.get("folds", [])

    if len(folds) != n_splits:
        raise ValueError(f"Expected {n_splits} folds, but found {len(folds)}.")

    manifest_uids: Set[str] = set(manifest_df["utterance_id"].astype(str))
    uid_to_speaker: Dict[str, str] = dict(zip(manifest_df["utterance_id"].astype(str), manifest_df[group_col].astype(str)))
    uid_to_label: Dict[str, int] = dict(zip(manifest_df["utterance_id"].astype(str), manifest_df["sarcasm"].astype(int)))

    all_test_uids: List[str] = []

    for fold in folds:
        f_idx = fold["fold"]
        train_uids = set(fold["train_utterance_ids"])
        test_uids = set(fold["test_utterance_ids"])

        # Check disjointness of utterance IDs within the fold
        id_overlap = train_uids.intersection(test_uids)
        if id_overlap:
            raise ValueError(f"Utterance ID overlap in fold {f_idx}: {id_overlap}")

        all_test_uids.extend(list(test_uids))

        # Check speaker leakage using ground truth speakers from manifest
        train_spks = {uid_to_speaker[u] for u in train_uids if u in uid_to_speaker}
        test_spks = {uid_to_speaker[u] for u in test_uids if u in uid_to_speaker}
        spk_leakage = train_spks.intersection(test_spks)
        if spk_leakage:
            raise ValueError(f"Speaker leakage in fold {f_idx}! Overlapping speakers: {spk_leakage}")

        # Check class presence
        test_labels = {uid_to_label[u] for u in test_uids if u in uid_to_label}
        if len(test_labels) < 2:
            raise ValueError(f"Fold {f_idx} test set lacks both classes! Classes present: {test_labels}")

    # Check that all test sets partition the dataset exactly
    if set(all_test_uids) != manifest_uids:
        missing = manifest_uids - set(all_test_uids)
        extra = set(all_test_uids) - manifest_uids
        raise ValueError(f"Folds do not partition manifest! Missing: {len(missing)}, Extra: {len(extra)}")

    if len(all_test_uids) != len(manifest_uids):
        raise ValueError("Duplicate samples assigned across test folds!")

    summary = {
        "status": "VALID",
        "n_splits": n_splits,
        "total_samples": len(manifest_uids),
        "fold_test_counts": [f["test_sample_count"] for f in folds],
        "zero_speaker_leakage": True,
    }
    logger.info("Fold validation passed successfully: zero speaker leakage across all %d folds.", n_splits)
    return summary


def save_frozen_folds(
    folds_data: Dict[str, Any],
    output_path: Union[str, Path] = DEFAULT_FOLDS_PATH,
) -> Path:
    """Save and freeze fold assignments to a JSON file.

    Args:
        folds_data: Dictionary of fold assignments.
        output_path: Target path (default: 'data/folds.json').

    Returns:
        Path object of saved file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(folds_data, f, indent=2)
    logger.info("Saved frozen folds to %s", path)
    return path


def load_frozen_folds(
    folds_path: Union[str, Path] = DEFAULT_FOLDS_PATH,
    manifest_path: Optional[Union[str, Path]] = DEFAULT_MANIFEST_PATH,
) -> Dict[str, Any]:
    """Load and validate the frozen fold assignments from disk.

    Args:
        folds_path: Path to folds.json.
        manifest_path: Optional path to manifest.csv to validate against.

    Returns:
        Validated folds data dictionary.
    """
    path = Path(folds_path)
    if not path.is_file():
        raise FileNotFoundError(f"Folds file not found at {path}. Generate it using generate_and_freeze_folds().")

    with open(path, "r", encoding="utf-8") as f:
        folds_data = json.load(f)

    if manifest_path:
        m_path = Path(manifest_path)
        if m_path.is_file():
            manifest_df = pd.read_csv(m_path)
            validate_folds_against_manifest(folds_data, manifest_df)

    return folds_data


def get_fold_assignments(
    manifest_df: pd.DataFrame,
    folds_data: Dict[str, Any],
) -> pd.Series:
    """Map each sample in the manifest to its assigned test fold (0 to n_splits - 1).

    Args:
        manifest_df: Manifest DataFrame.
        folds_data: Loaded folds data dictionary.

    Returns:
        Series of fold integers indexed by utterance_id.
    """
    uid_to_fold: Dict[str, int] = {}
    for fold in folds_data["folds"]:
        f_idx = fold["fold"]
        for uid in fold["test_utterance_ids"]:
            uid_to_fold[str(uid)] = f_idx

    assignments = manifest_df["utterance_id"].astype(str).map(uid_to_fold)
    return assignments


def get_fold_split(
    manifest_df: pd.DataFrame,
    folds_data: Dict[str, Any],
    fold_idx: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Retrieve train and test DataFrames for a specific fold index.

    Args:
        manifest_df: Master manifest DataFrame.
        folds_data: Loaded folds data dictionary.
        fold_idx: Fold index (0 to n_splits - 1).

    Returns:
        Tuple of (train_df, test_df).
    """
    folds = {f["fold"]: f for f in folds_data["folds"]}
    if fold_idx not in folds:
        raise ValueError(f"Fold index {fold_idx} not found. Available folds: {sorted(folds.keys())}")

    target_fold = folds[fold_idx]
    train_uids = set(target_fold["train_utterance_ids"])
    test_uids = set(target_fold["test_utterance_ids"])

    manifest_uid_col = manifest_df["utterance_id"].astype(str)
    train_df = manifest_df[manifest_uid_col.isin(train_uids)].copy().reset_index(drop=True)
    test_df = manifest_df[manifest_uid_col.isin(test_uids)].copy().reset_index(drop=True)

    return train_df, test_df


def generate_and_freeze_folds(
    manifest_path: Union[str, Path] = DEFAULT_MANIFEST_PATH,
    output_path: Union[str, Path] = DEFAULT_FOLDS_PATH,
    n_splits: int = N_SPLITS,
) -> Dict[str, Any]:
    """High-level runner: loads manifest, generates folds, validates, and freezes to disk.

    Args:
        manifest_path: Path to manifest.csv.
        output_path: Path to save folds.json.
        n_splits: Number of folds (default: 5).

    Returns:
        Generated folds data dictionary.
    """
    manifest_df = pd.read_csv(manifest_path)
    folds_data = generate_speaker_folds(manifest_df, n_splits=n_splits)
    validate_folds_against_manifest(folds_data, manifest_df)
    save_frozen_folds(folds_data, output_path=output_path)
    return folds_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    repo_root = Path(__file__).resolve().parents[3]
    m_csv = repo_root / "data" / "manifest.csv"
    f_json = repo_root / "data" / "folds.json"
    generate_and_freeze_folds(manifest_path=m_csv, output_path=f_json)

