"""Unit tests for RoBERTa feature extraction and classification head."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from sarcastone.text.embed import (
    EMBEDDING_DIM,
    SarcasmClassifierHead,
    mean_pooling,
    train_eval_head_fold,
)


class TestRobertaEmbeddings(unittest.TestCase):
    """Test suite for attention-mask-aware pooling and classification head."""

    def test_mean_pooling_attention_mask_invariance(self):
        """Verify that padding tokens (mask=0) contribute zero to pooled embedding."""
        hidden_dim = 16
        seq_len_1 = 4
        seq_len_2 = 8

        # Create identical hidden states for first 4 tokens
        torch.manual_seed(42)
        h_tokens = torch.randn(1, seq_len_1, hidden_dim)

        # Unpadded sequence (length 4)
        mask_unpadded = torch.ones(1, seq_len_1)
        pooled_unpadded = mean_pooling(h_tokens, mask_unpadded)

        # Padded sequence: same 4 tokens + 4 arbitrary padding tokens
        h_padded = torch.cat([h_tokens, torch.randn(1, seq_len_2 - seq_len_1, hidden_dim)], dim=1)
        mask_padded = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float32)
        pooled_padded = mean_pooling(h_padded, mask_padded)

        # Assert exact equivalence: padding tokens must have 0 effect
        np.testing.assert_allclose(
            pooled_unpadded.detach().numpy(),
            pooled_padded.detach().numpy(),
            rtol=1e-5,
            atol=1e-6,
            err_msg="Padding tokens altered the pooled representation!",
        )

    def test_classifier_head_forward_shape(self):
        """Test SarcasmClassifierHead output dimensions."""
        batch_size = 8
        head = SarcasmClassifierHead(input_dim=EMBEDDING_DIM, hidden_dim=64, num_classes=2)
        x = torch.randn(batch_size, EMBEDDING_DIM)
        logits = head(x)
        self.assertEqual(logits.shape, (batch_size, 2))

    def test_train_eval_head_fold_deterministic(self):
        """Test single-fold training function executes and returns valid metrics."""
        np.random.seed(42)
        X_train = np.random.randn(40, EMBEDDING_DIM).astype(np.float32)
        y_train = np.random.randint(0, 2, size=40)
        X_test = np.random.randn(10, EMBEDDING_DIM).astype(np.float32)
        y_test = np.random.randint(0, 2, size=10)

        metrics, preds, probs = train_eval_head_fold(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            seed=42,
            epochs=5,
            batch_size=16,
            hidden_dim=32,
            device=torch.device("cpu"),
        )

        self.assertIn("accuracy", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertEqual(len(preds), 10)
        self.assertEqual(len(probs), 10)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))

    def test_exported_features_exist_and_shape(self):
        """Test that exported text features match expected dimensions and dtype."""
        feat_dir = Path(__file__).resolve().parents[1] / "features" / "text"
        if not feat_dir.exists():
            self.skipTest("Features directory not yet generated.")

        npz_files = list(feat_dir.glob("*.npz"))
        self.assertGreater(len(npz_files), 0, "No .npz archives found in features/text")

        # Test loading one .npy file
        npy_files = list(feat_dir.glob("*.npy"))
        self.assertGreater(len(npy_files), 0, "No .npy files found in features/text")

        sample_npy = np.load(npy_files[0])
        self.assertEqual(sample_npy.shape, (768,))
        self.assertEqual(sample_npy.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()

