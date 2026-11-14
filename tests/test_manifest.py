"""Unit tests for sarcastone.data.manifest."""

import json
import unittest
from pathlib import Path

import pandas as pd

from sarcastone.data.manifest import (
    CORRUPTED_KEY_RAW,
    CORRUPTED_KEY_SCENE,
    TURN_DELIMITER,
    build_and_save_manifest,
    load_raw_mustard,
    parse_context_turn,
    validate_manifest,
)


class TestManifest(unittest.TestCase):
    """Test suite for manifest loading, extraction, and validation."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.raw_csv = cls.repo_root / "data" / "mustard++_text.csv"
        cls.manifest_csv = cls.repo_root / "data" / "manifest.csv"
        cls.utterances_csv = cls.repo_root / "data" / "manifest_utterances.csv"

        if not cls.manifest_csv.is_file():
            build_and_save_manifest(raw_csv_path=cls.raw_csv, output_dir=cls.repo_root / "data")

        cls.manifest_df = pd.read_csv(cls.manifest_csv)
        cls.utterances_df = pd.read_csv(cls.utterances_csv)

    def test_parse_context_turn(self):
        """Test extraction of integer turn indices from context keys."""
        # Zero-padded (original MUStARD format)
        self.assertEqual(parse_context_turn("1_10004_c_00"), 0)
        self.assertEqual(parse_context_turn("1_10004_c_09"), 9)
        self.assertEqual(parse_context_turn("2_369_c_10"), 10)

        # Unpadded (MUStARD++ added format)
        self.assertEqual(parse_context_turn("1_S09E01_027_c_0"), 0)
        self.assertEqual(parse_context_turn("1_S09E01_027_c_5"), 5)
        self.assertEqual(parse_context_turn("3_S04E05_320_c_12"), 12)

        # Non-context keys
        self.assertIsNone(parse_context_turn("1_10004_u"))
        self.assertIsNone(parse_context_turn("Disgust"))
        self.assertIsNone(parse_context_turn(""))

    def test_sample_manifest_counts_and_balance(self):
        """Test that sample manifest contains exactly 1,202 samples with 50/50 balance."""
        self.assertEqual(len(self.manifest_df), 1202)
        self.assertEqual(self.manifest_df["utterance_id"].nunique(), 1202)
        self.assertEqual(self.manifest_df["scene_id"].nunique(), 1202)

        sarcasm_counts = self.manifest_df["sarcasm"].value_counts().to_dict()
        self.assertEqual(sarcasm_counts.get(0), 601)
        self.assertEqual(sarcasm_counts.get(1), 601)

    def test_utterance_manifest_counts(self):
        """Test utterance registry contains all 6,041 utterances (1,202 targets, 4,839 contexts)."""
        self.assertEqual(len(self.utterances_df), 6041)
        self.assertEqual(self.utterances_df["is_target"].sum(), 1202)
        self.assertEqual((~self.utterances_df["is_target"]).sum(), 4839)

    def test_no_critical_nulls(self):
        """Ensure no null values in critical manifest columns."""
        critical_cols = [
            "utterance_id", "scene_id", "target_key", "raw_target_key",
            "target_text", "target_speaker", "show", "end_time",
            "sarcasm", "sarcasm_type", "implicit_emotion", "explicit_emotion",
            "valence", "arousal", "context_turn_count", "context_text",
            "context_turns_json", "context_speakers", "has_text", "has_audio", "has_video"
        ]
        for col in critical_cols:
            self.assertEqual(
                self.manifest_df[col].isnull().sum(), 0,
                f"Unexpected null values found in column '{col}'"
            )

    def test_context_ordering_and_gap_preservation(self):
        """Verify context turns are strictly in ascending order and gaps are preserved."""
        for _, row in self.manifest_df.iterrows():
            turns = json.loads(row["context_turns_json"])
            turn_indices = [t["turn"] for t in turns]
            self.assertEqual(turn_indices, sorted(turn_indices), f"Turns out of order in scene {row['scene_id']}")

        # Verify known scene with gaps: 1_S09E13_024 has turns [0, 1, 4, 5]
        gap_row = self.manifest_df[self.manifest_df["scene_id"] == "1_S09E13_024"].iloc[0]
        gap_turns = [t["turn"] for t in json.loads(gap_row["context_turns_json"])]
        self.assertEqual(gap_turns, [0, 1, 4, 5])

    def test_option_a_and_b_alignment(self):
        """Verify Option A (structured JSON) and Option B (serialized text) align."""
        sample_rows = self.manifest_df.sample(20, random_state=42)
        for _, row in sample_rows.iterrows():
            structured_turns = json.loads(row["context_turns_json"])
            serialized_text = row["context_text"]
            if structured_turns:
                reconstructed = TURN_DELIMITER.join(t["text"] for t in structured_turns)
                self.assertEqual(serialized_text, reconstructed)
            else:
                self.assertEqual(serialized_text, "")

    def test_anomaly_resolutions(self):
        """Verify that known dataset anomalies are resolved with 100% provenance."""
        # Row 2596: target key 'Disgust' in scene '1_S11E03_067'
        corrupted_row = self.manifest_df[self.manifest_df["scene_id"] == CORRUPTED_KEY_SCENE].iloc[0]
        self.assertEqual(corrupted_row["raw_target_key"], CORRUPTED_KEY_RAW)
        self.assertEqual(corrupted_row["target_key"], f"{CORRUPTED_KEY_SCENE}_u")
        self.assertEqual(corrupted_row["target_speaker"], "BERNADETTE")

        # Row 613: SHOW='0' in scene '1_2837'
        show_row = self.manifest_df[self.manifest_df["scene_id"] == "1_2837"].iloc[0]
        self.assertEqual(show_row["show"], "BBT")

    def test_modality_flags(self):
        """Verify modality availability flags reflect current repository state."""
        self.assertTrue((self.manifest_df["has_text"] == True).all())
        # Audio and video are not downloaded yet
        self.assertTrue((self.manifest_df["has_audio"] == False).all())
        self.assertTrue((self.manifest_df["has_video"] == False).all())


if __name__ == "__main__":
    unittest.main()

