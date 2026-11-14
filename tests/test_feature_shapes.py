"""Feature shape and integrity tests for SarcasTone text branch."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class TestFeatureShapes(unittest.TestCase):
    """Test suite ensuring all exported features adhere to contracts."""

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.manifest_path = cls.repo_root / "data" / "manifest.csv"
        cls.feat_dir = cls.repo_root / "features" / "text"
        cls.manifest_df = pd.read_csv(cls.manifest_path)

    def test_manifest_sample_count(self):
        """Ensure manifest has exactly 1202 target samples."""
        self.assertEqual(len(self.manifest_df), 1202)

    def test_text_feature_shapes_and_values(self):
        """Check all exported utterance embeddings match shape (768,) and have valid values."""
        if not self.feat_dir.exists():
            self.skipTest("Feature directory does not exist.")

        uids = self.manifest_df["utterance_id"].astype(str).tolist()
        missing = []
        invalid_shape = []
        has_nan_or_inf = []

        for uid in uids:
            npy_path = self.feat_dir / f"{uid}.npy"
            if not npy_path.exists():
                missing.append(uid)
                continue

            arr = np.load(npy_path)
            if arr.shape != (768,):
                invalid_shape.append((uid, arr.shape))
            if not np.all(np.isfinite(arr)):
                has_nan_or_inf.append(uid)

        self.assertEqual(len(missing), 0, f"Missing .npy features for {len(missing)} utterances: {missing[:5]}")
        self.assertEqual(len(invalid_shape), 0, f"Incorrect shape for: {invalid_shape[:5]}")
        self.assertEqual(len(has_nan_or_inf), 0, f"Found NaN/Inf in: {has_nan_or_inf[:5]}")

    def test_npz_archive_integrity(self):
        """Check the compressed .npz archive contains all 1202 utterances."""
        npz_path = self.feat_dir / "roberta_embeddings_utterance_only.npz"
        if not npz_path.exists():
            self.skipTest("npz archive not found.")

        data = np.load(npz_path)
        uids = set(self.manifest_df["utterance_id"].astype(str))
        archive_keys = set(data.files)

        self.assertTrue(uids.issubset(archive_keys), "Not all manifest utterances present in .npz archive")
        sample_vec = data[next(iter(archive_keys))]
        self.assertEqual(sample_vec.shape, (768,))


if __name__ == "__main__":
    unittest.main()

