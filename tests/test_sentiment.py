"""Unit tests for pre-trained 3-class sentiment model and feature exports."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sarcastone.text.sentiment import (
    IDX_TO_CLASS,
    SENTIMENT_CLASSES,
    SentimentFeatureExtractor,
    validate_sentiment_against_valence,
)


class TestSentimentFeatures(unittest.TestCase):
    """Test suite for sentiment feature extraction and export."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.manifest_path = cls.repo_root / "data" / "manifest.csv"
        cls.emotion_dir = cls.repo_root / "features" / "emotion"

    def test_sentiment_extractor_output_contract(self):
        """Test probability shapes, ranges, and normalization."""
        texts = [
            "This is wonderful news, I am thrilled!",
            "It is completely standard and ordinary.",
            "This is horrific and totally ruined my day.",
        ]
        extractor = SentimentFeatureExtractor()
        probs = extractor.predict_probabilities(texts)

        self.assertEqual(probs.shape, (3, 3))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))
        # Each row must sum to 1.0
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(3), rtol=1e-5, atol=1e-5)

        # Expected top sentiments for strong positive vs negative
        pred_pos = IDX_TO_CLASS[int(np.argmax(probs[0]))]
        pred_neg = IDX_TO_CLASS[int(np.argmax(probs[2]))]
        self.assertEqual(pred_pos, "positive")
        self.assertEqual(pred_neg, "negative")

    def test_exported_sentiment_files(self):
        """Verify exported files in features/emotion/."""
        if not self.emotion_dir.exists():
            self.skipTest("features/emotion not generated.")

        csv_path = self.emotion_dir / "sentiment_probabilities.csv"
        self.assertTrue(csv_path.exists())
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 1202)
        expected_cols = {"utterance_id", "prob_negative", "prob_neutral", "prob_positive", "predicted_sentiment"}
        self.assertTrue(expected_cols.issubset(set(df.columns)))

        # Check npz archive
        npz_path = self.emotion_dir / "sentiment_probabilities.npz"
        self.assertTrue(npz_path.exists())
        data = np.load(npz_path)
        self.assertEqual(len(data.files), 1202)
        sample_vec = data[next(iter(data.files))]
        self.assertEqual(sample_vec.shape, (3,))
        self.assertEqual(sample_vec.dtype, np.float32)

    def test_valence_validation_monotonicity(self):
        """Verify validation logic preserves monotonic valence trend."""
        sample_sentiment = pd.DataFrame({
            "utterance_id": ["1", "2", "3"],
            "prob_negative": [0.8, 0.1, 0.1],
            "prob_neutral": [0.1, 0.8, 0.1],
            "prob_positive": [0.1, 0.1, 0.8],
            "predicted_sentiment": ["negative", "neutral", "positive"],
        })
        sample_manifest = pd.DataFrame({
            "utterance_id": ["1", "2", "3"],
            "valence": [2.0, 4.0, 6.0],
            "arousal": [3.0, 3.0, 3.0],
            "sarcasm": [0, 0, 0],
        })
        rep = validate_sentiment_against_valence(sample_sentiment, sample_manifest)
        self.assertGreater(rep["pearson_correlation_with_valence"], 0.8)


if __name__ == "__main__":
    unittest.main()

