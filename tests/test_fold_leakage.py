"""Unit tests for speaker-independent cross-validation folds and leakage verification."""

import unittest
from pathlib import Path

import pandas as pd

from sarcastone.data.folds import (
    N_SPLITS,
    generate_speaker_folds,
    get_fold_assignments,
    get_fold_split,
    load_frozen_folds,
    validate_folds_against_manifest,
)


class TestFoldLeakage(unittest.TestCase):
    """Test suite ensuring strict speaker independence and zero data leakage across folds."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.manifest_csv = cls.repo_root / "data" / "manifest.csv"
        cls.folds_json = cls.repo_root / "data" / "folds.json"

        cls.manifest_df = pd.read_csv(cls.manifest_csv)
        cls.folds_data = load_frozen_folds(cls.folds_json, cls.manifest_csv)

    def test_n_splits_count(self):
        """Verify exactly 5 folds exist in the frozen assignments."""
        self.assertEqual(len(self.folds_data["folds"]), N_SPLITS)
        self.assertEqual(self.folds_data["metadata"]["n_splits"], N_SPLITS)

    def test_zero_speaker_leakage(self):
        """CRITICAL: Assert zero speaker overlap between train and test for all folds."""
        for fold in self.folds_data["folds"]:
            fold_idx = fold["fold"]
            train_speakers = set(fold["train_speakers"])
            test_speakers = set(fold["test_speakers"])

            overlap = train_speakers.intersection(test_speakers)
            self.assertEqual(
                len(overlap), 0,
                f"Speaker leakage in fold {fold_idx}! Overlapping speakers: {overlap}"
            )

    def test_exact_dataset_partitioning(self):
        """Assert all 1,202 samples are assigned to exactly one test fold."""
        manifest_uids = set(self.manifest_df["utterance_id"].astype(str))
        all_test_uids = []

        for fold in self.folds_data["folds"]:
            test_uids = fold["test_utterance_ids"]
            all_test_uids.extend(test_uids)

        # No duplicate test sample across folds
        self.assertEqual(len(all_test_uids), len(manifest_uids))
        # Union matches manifest exactly
        self.assertEqual(set(all_test_uids), manifest_uids)

    def test_fold_sample_counts_and_balance(self):
        """Verify uniform fold sizes (~240 samples) and class representation in each fold."""
        for fold in self.folds_data["folds"]:
            test_count = fold["test_sample_count"]
            # All folds should be close to 240 samples
            self.assertTrue(230 <= test_count <= 250, f"Fold count {test_count} outside expected range")

            sarc_count = fold["test_sarcastic_count"]
            nonsarc_count = fold["test_nonsarcastic_count"]
            self.assertEqual(sarc_count + nonsarc_count, test_count)

            # Both classes must be present and reasonably balanced
            self.assertTrue(sarc_count > 50, "Insufficient sarcastic samples in test fold")
            self.assertTrue(nonsarc_count > 50, "Insufficient non-sarcastic samples in test fold")

    def test_get_fold_assignments(self):
        """Verify get_fold_assignments creates a complete, valid mapping."""
        assignments = get_fold_assignments(self.manifest_df, self.folds_data)
        self.assertEqual(len(assignments), 1202)
        self.assertFalse(assignments.isnull().any())
        self.assertEqual(set(assignments.unique()), set(range(N_SPLITS)))

    def test_get_fold_split(self):
        """Verify train/test DataFrame generation for an arbitrary fold."""
        train_df, test_df = get_fold_split(self.manifest_df, self.folds_data, fold_idx=0)
        self.assertEqual(len(train_df) + len(test_df), 1202)
        self.assertEqual(len(test_df), 239)
        self.assertEqual(len(train_df), 963)

        train_speakers = set(train_df["target_speaker"])
        test_speakers = set(test_df["target_speaker"])
        self.assertEqual(len(train_speakers.intersection(test_speakers)), 0)


if __name__ == "__main__":
    unittest.main()

