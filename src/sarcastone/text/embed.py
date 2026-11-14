"""RoBERTa text embedding extraction and classification head for SarcasTone.

Implements the text branch model architecture specified in AGENTS.md Section 6 & 7 and
OKF docs/sarcastone_text_branch_okf/models/roberta.md:
- Uses pre-trained RoBERTa-base encoder kept strictly FROZEN.
- Computes attention-mask-aware mean (average) pooling:
    e = sum(h_i * m_i) / sum(m_i)
  guaranteeing padding tokens do not contribute to the representation.
- Trains a small classification head on top of the 768-dim pooled representations.
- Exports embeddings keyed by utterance_id to features/text/ for multimodal fusion.
- Evaluates across the frozen speaker-independent 5 folds (data/folds.json) and 3 seeds.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from sarcastone.data.folds import get_fold_split, load_frozen_folds
from sarcastone.eval.metrics import aggregate_fold_metrics, compute_classification_metrics
from sarcastone.text.preprocess import (
    UTTERANCE_ONLY,
    UTTERANCE_PLUS_CONTEXT,
    VALID_CONDITIONS,
    process_manifest,
)
from sarcastone.utils.seeds import EXPERIMENT_SEEDS, set_seed

logger = logging.getLogger(__name__)

ROBERTA_MODEL_NAME = "roberta-base"
EMBEDDING_DIM = 768
DEFAULT_MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 32


def mean_pooling(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute attention-mask-aware average (mean) pooled sentence embeddings.

    Applies the formula: e = sum(h_i * m_i) / sum(m_i)
    Padding tokens receive exactly zero weight and are excluded from the denominator.

    Args:
        last_hidden_state: Token embeddings tensor of shape [batch_size, seq_len, hidden_dim].
        attention_mask: Attention mask tensor of shape [batch_size, seq_len].

    Returns:
        Pooled embedding tensor of shape [batch_size, hidden_dim].
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class TextDataset(Dataset):
    """PyTorch Dataset for text tokenization."""

    def __init__(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }


class SarcasmClassifierHead(nn.Module):
    """Small classification head trained on top of frozen RoBERTa embeddings."""

    def __init__(
        self,
        input_dim: int = EMBEDDING_DIM,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class RobertaFeatureExtractor:
    """Extracts frozen, attention-mask-aware mean-pooled RoBERTa embeddings."""

    def __init__(
        self,
        model_name: str = ROBERTA_MODEL_NAME,
        device: Optional[torch.device] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size

        logger.info("Loading frozen %s on device: %s", model_name, self.device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

        # Freeze encoder completely (mandated by AGENTS.md Section 6)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)

    @torch.no_grad()
    def extract_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract attention-mask-aware mean-pooled embeddings for a list of texts.

        Args:
            texts: List of text strings.

        Returns:
            NumPy array of shape [len(texts), 768] with dtype float32.
        """
        dataset = TextDataset(texts, self.tokenizer, max_length=self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_embeddings: List[np.ndarray] = []
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
            all_embeddings.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(all_embeddings)


def export_roberta_embeddings(
    manifest_path: Union[str, Path] = "data/manifest.csv",
    output_dir: Union[str, Path] = "features/text",
    condition: str = UTTERANCE_ONLY,
    extractor: Optional[RobertaFeatureExtractor] = None,
) -> Dict[str, Path]:
    """Extract and export 768-dim .npy embeddings keyed by utterance_id for multimodal fusion.

    Args:
        manifest_path: Path to manifest.csv.
        output_dir: Directory where .npy files are saved.
        condition: 'utterance_only' or 'utterance_plus_context'.
        extractor: Optional existing RobertaFeatureExtractor instance.

    Returns:
        Dictionary mapping utterance_id to saved .npy file path.
    """
    manifest_df = pd.read_csv(manifest_path)
    samples = process_manifest(manifest_df, condition=condition)
    texts = [s.get_input_text(condition=condition) for s in samples]
    uids = [s.utterance_id for s in samples]

    ext = extractor or RobertaFeatureExtractor()
    embeddings = ext.extract_embeddings(texts)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    saved_paths: Dict[str, Path] = {}
    for uid, emb in zip(uids, embeddings):
        npy_path = out_p / f"{uid}.npy"
        np.save(npy_path, emb)
        saved_paths[uid] = npy_path

    # Also save an aggregated .npz archive for convenient single-file loading
    archive_path = out_p / f"roberta_embeddings_{condition}.npz"
    np.savez_compressed(archive_path, **{uid: emb for uid, emb in zip(uids, embeddings)})
    logger.info("Saved %d feature files and archive to %s", len(saved_paths), out_p)

    return saved_paths


def train_eval_head_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    hidden_dim: int = 128,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Train small classification head on frozen embeddings for a single fold.

    Args:
        X_train: Training embeddings [N_train, 768].
        y_train: Training labels [N_train].
        X_test: Test embeddings [N_test, 768].
        y_test: Test labels [N_test].
        seed: Random seed.
        epochs: Training epochs.
        lr: Learning rate.
        weight_decay: Weight decay.
        batch_size: Training batch size.
        hidden_dim: Hidden dimension of classification head.
        device: Torch device.

    Returns:
        Tuple of (metrics_dict, y_pred, y_prob).
    """
    set_seed(seed)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head = SarcasmClassifierHead(input_dim=X_train.shape[1], hidden_dim=hidden_dim).to(dev)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_tensor_x = torch.tensor(X_train, dtype=torch.float32)
    train_tensor_y = torch.tensor(y_train, dtype=torch.long)
    train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    head.train()
    for _ in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(dev), by.to(dev)
            optimizer.zero_grad()
            logits = head(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    head.eval()
    with torch.no_grad():
        test_tensor_x = torch.tensor(X_test, dtype=torch.float32).to(dev)
        logits = head(test_tensor_x)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    metrics = compute_classification_metrics(y_test, preds, y_prob=probs)
    return metrics, preds, probs


def run_roberta_cv_single_seed(
    manifest_df: pd.DataFrame,
    folds_data: Dict[str, Any],
    all_embeddings_dict: Dict[str, np.ndarray],
    condition: str = UTTERANCE_ONLY,
    seed: int = 42,
    epochs: int = 50,
) -> Dict[str, Any]:
    """Run 5-fold cross-validation for a single seed using cached RoBERTa embeddings."""
    set_seed(seed)
    fold_results: List[Dict[str, Any]] = []
    all_predictions: List[Dict[str, Any]] = []

    for fold in folds_data["folds"]:
        fold_idx = fold["fold"]
        train_df, test_df = get_fold_split(manifest_df, folds_data, fold_idx=fold_idx)

        train_uids = train_df["utterance_id"].astype(str).tolist()
        test_uids = test_df["utterance_id"].astype(str).tolist()

        X_train = np.array([all_embeddings_dict[uid] for uid in train_uids], dtype=np.float32)
        y_train = train_df["sarcasm"].values.astype(np.int64)

        X_test = np.array([all_embeddings_dict[uid] for uid in test_uids], dtype=np.float32)
        y_test = test_df["sarcasm"].values.astype(np.int64)

        metrics, y_pred, y_prob = train_eval_head_fold(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            seed=seed,
            epochs=epochs,
        )

        fold_record = {
            "fold": fold_idx,
            "seed": seed,
            "condition": condition,
            "test_sample_count": len(y_test),
            "test_speakers": fold["test_speakers"],
            "metrics": metrics,
        }
        fold_results.append(fold_record)

        for uid, yt, yp, prob in zip(test_uids, y_test, y_pred, y_prob):
            all_predictions.append({
                "utterance_id": uid,
                "fold": fold_idx,
                "seed": seed,
                "condition": condition,
                "y_true": int(yt),
                "y_pred": int(yp),
                "y_prob": round(float(prob), 4),
            })

    agg_metrics = aggregate_fold_metrics([fr["metrics"] for fr in fold_results])
    return {
        "seed": seed,
        "condition": condition,
        "aggregate_metrics": agg_metrics,
        "fold_results": fold_results,
        "predictions": all_predictions,
    }


def run_roberta_experiment(
    manifest_path: Union[str, Path] = "data/manifest.csv",
    folds_path: Union[str, Path] = "data/folds.json",
    seeds: Sequence[int] = EXPERIMENT_SEEDS,
    conditions: Sequence[str] = (UTTERANCE_ONLY, UTTERANCE_PLUS_CONTEXT),
    output_dir: Optional[Union[str, Path]] = "results",
    feature_dir: Optional[Union[str, Path]] = "features/text",
    epochs: int = 50,
) -> Dict[str, Any]:
    """Execute complete 3-seed × 5-fold RoBERTa frozen encoder experiment.

    Args:
        manifest_path: Path to manifest.csv.
        folds_path: Path to folds.json.
        seeds: Random seeds.
        conditions: Conditions to evaluate ('utterance_only', 'utterance_plus_context').
        output_dir: Target directory for result JSON.
        feature_dir: Target directory for cached .npy features.
        epochs: Epochs for training the head.

    Returns:
        Structured experiment report dictionary.
    """
    manifest_df = pd.read_csv(manifest_path)
    folds_data = load_frozen_folds(folds_path, manifest_path)

    extractor = RobertaFeatureExtractor()
    results_by_condition: Dict[str, Any] = {}

    for cond in conditions:
        logger.info("Extracting RoBERTa embeddings under condition: '%s'", cond)
        samples = process_manifest(manifest_df, condition=cond)
        texts = [s.get_input_text(condition=cond) for s in samples]
        uids = [s.utterance_id for s in samples]

        embeddings = extractor.extract_embeddings(texts)
        emb_dict = {uid: emb for uid, emb in zip(uids, embeddings)}

        # Export features to disk if feature_dir provided
        if feature_dir:
            out_p = Path(feature_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            for uid, emb in emb_dict.items():
                np.save(out_p / f"{uid}_{cond}.npy", emb)
            np.savez_compressed(out_p / f"roberta_embeddings_{cond}.npz", **emb_dict)

        seed_runs: List[Dict[str, Any]] = []
        for seed in seeds:
            logger.info("Evaluating RoBERTa head on 5 folds for seed: %d", seed)
            run = run_roberta_cv_single_seed(
                manifest_df=manifest_df,
                folds_data=folds_data,
                all_embeddings_dict=emb_dict,
                condition=cond,
                seed=seed,
                epochs=epochs,
            )
            seed_runs.append(run)

        seed_accs = [sr["aggregate_metrics"]["accuracy_mean"] for sr in seed_runs]
        seed_f1s = [sr["aggregate_metrics"]["f1_macro_mean"] for sr in seed_runs]
        seed_f1_weighted = [sr["aggregate_metrics"]["f1_weighted_mean"] for sr in seed_runs]

        overall_summary = {
            "accuracy_mean": round(float(np.mean(seed_accs)), 4),
            "accuracy_std": round(float(np.std(seed_accs)), 4),
            "f1_macro_mean": round(float(np.mean(seed_f1s)), 4),
            "f1_macro_std": round(float(np.std(seed_f1s)), 4),
            "f1_weighted_mean": round(float(np.mean(seed_f1_weighted)), 4),
            "f1_weighted_std": round(float(np.std(seed_f1_weighted)), 4),
        }

        results_by_condition[cond] = {
            "overall_summary": overall_summary,
            "seed_runs": seed_runs,
        }

        logger.info(
            "RoBERTa '%s' completed: Accuracy = %.4f ± %.4f, Macro F1 = %.4f ± %.4f",
            cond,
            overall_summary["accuracy_mean"],
            overall_summary["accuracy_std"],
            overall_summary["f1_macro_mean"],
            overall_summary["f1_macro_std"],
        )

    experiment_report = {
        "metadata": {
            "model": "RoBERTa-base (Frozen) + SarcasmClassifierHead",
            "pooling": "Attention-Mask-Aware Average Pooling",
            "embedding_dim": EMBEDDING_DIM,
            "protocol": "Speaker-Independent 5-Fold Cross-Validation",
            "seeds": list(seeds),
            "conditions": list(conditions),
            "n_folds": len(folds_data["folds"]),
            "total_samples": len(manifest_df),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "results": results_by_condition,
    }

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        res_file = out_p / "roberta_experiment_results.json"
        with open(res_file, "w", encoding="utf-8") as f:
            json.dump(experiment_report, f, indent=2)
        logger.info("Saved RoBERTa experiment report to %s", res_file)

    return experiment_report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    repo_root = Path(__file__).resolve().parents[3]
    m_path = repo_root / "data" / "manifest.csv"
    f_path = repo_root / "data" / "folds.json"
    r_dir = repo_root / "results"
    feat_dir = repo_root / "features" / "text"
    run_roberta_experiment(
        manifest_path=m_path,
        folds_path=f_path,
        output_dir=r_dir,
        feature_dir=feat_dir,
    )
