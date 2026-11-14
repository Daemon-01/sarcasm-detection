"""Unit tests for sarcastone.text.tfidf baseline."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sarcastone.data.folds import load_frozen_folds
from sarcastone.text.tfidf import (
    build_tfidf_vectorizer,
    run_tfidf_cv_single_seed,
    train_eval_tfidf_fold,
)


class TestTfidfBaseline(unittest.TestCase):
    """Test suite for TF-IDF + Logistic Regression baseline."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.manifest_csv = cls.repo_root / "data" / "manifest.csv"
        cls.folds_json = cls.repo_root / "data" / "folds.json"
        cls.manifest_df = pd.read_csv(cls.manifest_csv)
        cls.folds_data = load_frozen_folds(cls.folds_json, cls.manifest_csv)

    def test_vectorizer_creation(self):
        """Verify vectorizer configuration parameters."""
        vec = build_tfidf_vectorizer(max_features=1000)
        self.assertEqual(vec.max_features, 1000)
        self.assertEqual(vec.ngram_range, (1, 2))
        self.assertTrue(vec.sublinear_tf)

    def test_single_fold_training_leakage_free(self):
        """Verify single fold training executes cleanly without leakage."""
        train_texts = ["This is great", "I love this so much", "Terrible day", "Awful news"]
        train_labels = np.array([1, 1, 0, 0])
        test_texts = ["This is great news", "Awful day"]
        test_labels = np.array([1, 0])

        metrics, y_pred, y_prob = train_eval_tfidf_fold(
            train_texts=train_texts,
            train_labels=train_labels,
            test_texts=test_texts,
            test_labels=test_labels,
            seed=42,
        )
        self.assertEqual(len(y_pred), 2)
        self.assertEqual(len(y_prob), 2)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1_macro", metrics)

    def test_cross_validation_single_seed(self):
        """Verify 5-fold CV run on full dataset returns complete fold metrics."""
        results = run_tfidf_cv_single_seed(
            manifest_df=self.manifest_df,
            folds_data=self.folds_data,
            condition="utterance_only",
            seed=42,
        )
        self.assertEqual(len(results["fold_results"]), 5)
        self.assertEqual(len(results["predictions"]), 1202)
        self.assertIn("accuracy_mean", results["aggregate_metrics"])
        self.assertIn("f1_macro_mean", results["aggregate_metrics"])


if __name__ == "__main__":
    unittest.main()

