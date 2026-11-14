"""Unit tests for classification metrics and aggregation."""

import unittest

import numpy as np

from sarcastone.eval.metrics import (
    aggregate_fold_metrics,
    compute_classification_metrics,
    format_metrics_summary,
)


class TestMetrics(unittest.TestCase):
    """Test suite for evaluation metric computations."""

    def test_perfect_predictions(self):
        """Verify metric calculation on perfect binary predictions."""
        y_true = [0, 1, 0, 1, 1, 0]
        y_pred = [0, 1, 0, 1, 1, 0]
        metrics = compute_classification_metrics(y_true, y_pred)

        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1_macro"], 1.0)
        self.assertEqual(metrics["precision_macro"], 1.0)
        self.assertEqual(metrics["recall_macro"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], {"tn": 3, "fp": 0, "fn": 0, "tp": 3})

    def test_imperfect_predictions(self):
        """Verify metric calculation on known imperfect predictions."""
        y_true = [1, 1, 1, 0, 0, 0]
        y_pred = [1, 1, 0, 0, 0, 1]
        metrics = compute_classification_metrics(y_true, y_pred)

        # TP=2, FN=1, TN=2, FP=1 -> Accuracy = 4/6 = 0.6667
        self.assertAlmostEqual(metrics["accuracy"], 0.6667, places=3)
        self.assertEqual(metrics["confusion_matrix"]["tp"], 2)
        self.assertEqual(metrics["confusion_matrix"]["fn"], 1)
        self.assertEqual(metrics["confusion_matrix"]["tn"], 2)
        self.assertEqual(metrics["confusion_matrix"]["fp"], 1)

    def test_with_probabilities(self):
        """Verify ROC AUC calculation when probabilities are supplied."""
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        y_prob = [0.1, 0.2, 0.8, 0.9]
        metrics = compute_classification_metrics(y_true, y_pred, y_prob=y_prob)

        self.assertIn("roc_auc", metrics)
        self.assertEqual(metrics["roc_auc"], 1.0)

    def test_aggregation(self):
        """Verify cross-fold aggregation calculates mean and standard deviation."""
        fold_metrics = [
            {"accuracy": 0.60, "f1_macro": 0.58, "precision_macro": 0.59, "recall_macro": 0.58,
             "f1_weighted": 0.60, "precision_weighted": 0.60, "recall_weighted": 0.60,
             "f1_sarcastic": 0.58, "precision_sarcastic": 0.59, "recall_sarcastic": 0.58},
            {"accuracy": 0.70, "f1_macro": 0.68, "precision_macro": 0.69, "recall_macro": 0.68,
             "f1_weighted": 0.70, "precision_weighted": 0.70, "recall_weighted": 0.70,
             "f1_sarcastic": 0.68, "precision_sarcastic": 0.69, "recall_sarcastic": 0.68},
        ]
        agg = aggregate_fold_metrics(fold_metrics)
        self.assertEqual(agg["accuracy_mean"], 0.65)
        self.assertEqual(agg["f1_macro_mean"], 0.63)
        self.assertEqual(agg["accuracy_std"], 0.05)


if __name__ == "__main__":
    unittest.main()

