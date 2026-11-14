"""Unit tests for sarcastone.text.preprocess."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sarcastone.text.preprocess import (
    TURN_DELIMITER,
    UTTERANCE_ONLY,
    UTTERANCE_PLUS_CONTEXT,
    TextSample,
    deserialize_context,
    get_text_inputs_and_labels,
    normalize_whitespace,
    prepare_context_turn,
    prepare_target_text,
    process_manifest,
    process_sample,
    serialize_context,
)


class TestPreprocess(unittest.TestCase):
    """Test suite for text preprocessing and context serialization."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.manifest_csv = cls.repo_root / "data" / "manifest.csv"
        cls.manifest_df = pd.read_csv(cls.manifest_csv)

    def test_normalize_whitespace_preserves_casing_and_punctuation(self):
        """Verify casing, punctuation, and wording are preserved while whitespace is normalized."""
        raw_text = "  Wait,   Sheldon!! \r\n Are you   serious??? \t That's brilliant!  "
        expected = "Wait, Sheldon!! Are you serious??? That's brilliant!"
        result = normalize_whitespace(raw_text)
        self.assertEqual(result, expected)

        # Check all caps preserved
        all_caps = "OH OF COURSE, WHAT A GENIUS."
        self.assertEqual(normalize_whitespace(all_caps), all_caps)

        # Check special punctuation preserved
        punct = "...Yes?! No... Wait--what?!"
        self.assertEqual(normalize_whitespace(punct), punct)

    def test_prepare_context_turn_speaker_options(self):
        """Test single context turn preparation with and without speaker prefix."""
        text = "  Hello   world!  "
        self.assertEqual(prepare_context_turn(text), "Hello world!")
        self.assertEqual(
            prepare_context_turn(text, speaker="SHELDON", include_speaker=True),
            "SHELDON: Hello world!"
        )
        # Empty or dash speaker should not prepend
        self.assertEqual(
            prepare_context_turn(text, speaker="-", include_speaker=True),
            "Hello world!"
        )

    def test_serialize_and_deserialize_context(self):
        """Test converting structured turns to Option B string and back."""
        turns = [
            {"turn": 0, "speaker": "LEONARD", "text": "Are you going to work?"},
            {"turn": 1, "speaker": "SHELDON", "text": "No, I am contemplating existence."},
            {"turn": 4, "speaker": "LEONARD", "text": "Enjoy your morning."},
        ]
        serialized = serialize_context(turns)
        expected = "Are you going to work? [TURN] No, I am contemplating existence. [TURN] Enjoy your morning."
        self.assertEqual(serialized, expected)

        # Deserialize
        deserialized = deserialize_context(serialized)
        self.assertEqual(deserialized, [
            "Are you going to work?",
            "No, I am contemplating existence.",
            "Enjoy your morning."
        ])

    def test_text_sample_conditions(self):
        """Test TextSample model input under both experimental conditions."""
        sample = TextSample(
            utterance_id="1_10004",
            target_text="A demented sex pervert?",
            context_text="Who was he? [TURN] A friend from church.",
            condition=UTTERANCE_ONLY,
            label=1,
        )
        # Utterance only
        self.assertEqual(sample.get_input_text(UTTERANCE_ONLY), "A demented sex pervert?")
        # Utterance plus context
        combined = sample.get_input_text(UTTERANCE_PLUS_CONTEXT)
        self.assertEqual(combined, "Who was he? [TURN] A friend from church. </s> A demented sex pervert?")

        # Invalid condition raises ValueError
        with self.assertRaises(ValueError):
            sample.get_input_text("invalid_condition")

    def test_process_manifest_on_actual_dataset(self):
        """Test batch preprocessing across all 1,202 samples from manifest.csv."""
        samples_utt = process_manifest(self.manifest_df, condition=UTTERANCE_ONLY)
        self.assertEqual(len(samples_utt), 1202)

        # Check sample fields populated
        first = samples_utt[0]
        self.assertTrue(len(first.target_text) > 0)
        self.assertIn(first.label, {0, 1})
        self.assertTrue(len(first.utterance_id) > 0)

        # Check utterance_plus_context condition
        samples_ctx = process_manifest(self.manifest_df, condition=UTTERANCE_PLUS_CONTEXT)
        self.assertEqual(len(samples_ctx), 1202)

        # Verify combined text format
        first_ctx = samples_ctx[0]
        self.assertIn("</s>", first_ctx.get_input_text())

    def test_get_text_inputs_and_labels(self):
        """Test array generation for downstream model vectorizers and classifiers."""
        texts, labels, uids = get_text_inputs_and_labels(self.manifest_df, condition=UTTERANCE_ONLY)
        self.assertEqual(len(texts), 1202)
        self.assertEqual(len(labels), 1202)
        self.assertEqual(len(uids), 1202)

        # Check label balance
        self.assertEqual((labels == 0).sum(), 601)
        self.assertEqual((labels == 1).sum(), 601)


if __name__ == "__main__":
    unittest.main()

