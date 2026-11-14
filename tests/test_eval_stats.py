"""Unit tests for evaluation runners, subset analysis, and statistical tests."""

import unittest
import numpy as np
import pandas as pd

from sarcastone.eval.stats import compute_paired_comparison, format_comparison_table
from sarcastone.eval.subsets import evaluate_sentiment_incongruence


class TestEvalStatsAndSubsets(unittest.TestCase):
    """Test suite for statistical testing and subset evaluation."""

    def test_paired_comparison_computation(self):
        """Test paired t-test returns expected keys and flags significant difference."""
        scores_a = [0.54, 0.55, 0.52, 0.56, 0.53]  # Baseline
        scores_b = [0.61, 0.59, 0.60, 0.63, 0.58]  # Stronger model
        res = compute_paired_comparison(scores_a, scores_b, "Baseline", "Model B", "accuracy")

        self.assertEqual(res["metric"], "accuracy")
        self.assertAlmostEqual(res["mean_difference"], 0.062, places=4)
        self.assertTrue(res["statistically_significant_p05"])
        self.assertLess(res["p_value"], 0.01)

        # Markdown table formatting
        table = format_comparison_table([res])
        self.assertIn("Baseline", table)
        self.assertIn("Model B", table)
        self.assertIn("+0.0620", table)

    def test_sentiment_incongruence_evaluation(self):
        """Test incongruent subset correctly splits and calculates metrics."""
        pred_df = pd.DataFrame({
            "utterance_id": ["u1", "u2", "u3", "u4"],
            "y_true": [1, 0, 1, 0],
            "y_pred": [0, 0, 1, 0],
        })
        manifest_df = pd.DataFrame({
            "utterance_id": ["u1", "u2", "u3", "u4"],
            "valence": [2.0, 4.0, 3.0, 5.0],
        })
        sent_df = pd.DataFrame({
            "utterance_id": ["u1", "u2", "u3", "u4"],
            "predicted_sentiment": ["positive", "neutral", "negative", "positive"],
        })
        # u1: positive sentiment + sarcasm 1 -> incongruent (y_pred=0 -> error)
        # u2: neutral sentiment + sarcasm 0 -> congruent (y_pred=0 -> correct)
        # u3: negative sentiment + sarcasm 1 -> congruent (y_pred=1 -> correct)
        # u4: positive sentiment + sarcasm 0 -> congruent (y_pred=0 -> correct)
        res = evaluate_sentiment_incongruence(pred_df, manifest_df, sent_df)

        self.assertEqual(res["total_evaluated"], 4)
        self.assertEqual(res["incongruent_subset"]["sample_count"], 1)
        self.assertEqual(res["incongruent_subset"]["accuracy"], 0.0)
        self.assertEqual(res["congruent_subset"]["sample_count"], 3)
        self.assertEqual(res["congruent_subset"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
