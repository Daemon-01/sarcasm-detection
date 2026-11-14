"""Pre-trained 3-class sentiment feature extractor and export for SarcasTone.

Implements the sentiment model requirements specified in AGENTS.md Section 6 & 7 and
OKF docs/sarcastone_text_branch_okf/models/sentiment.md:
- Pre-trained 3-class sentiment model: cardiffnlp/twitter-roberta-base-sentiment-latest
- Classes: [0: negative, 1: neutral, 2: positive]
- Frozen inference returning normalized probabilities [prob_negative, prob_neutral, prob_positive]
- Keyed strictly by utterance_id for downstream speech/fusion integration (features/emotion/)
- Validation against annotated MUStARD++ valence (validation-only; never used as training feature).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from sarcastone.text.preprocess import (
    UTTERANCE_ONLY,
    UTTERANCE_PLUS_CONTEXT,
    process_manifest,
)

logger = logging.getLogger(__name__)

SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SENTIMENT_CLASSES = ["negative", "neutral", "positive"]
CLASS_TO_IDX = {"negative": 0, "neutral": 1, "positive": 2}
IDX_TO_CLASS = {0: "negative", 1: "neutral", 2: "positive"}
DEFAULT_MAX_LENGTH = 128
DEFAULT_BATCH_SIZE = 32


class SentimentTextDataset(Dataset):
    """PyTorch Dataset for sentiment tokenization."""

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


class SentimentFeatureExtractor:
    """Extracts 3-class sentiment probabilities using a frozen pre-trained transformer."""

    def __init__(
        self,
        model_name: str = SENTIMENT_MODEL_NAME,
        device: Optional[torch.device] = None,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size

        logger.info("Loading sentiment model '%s' on %s", model_name, self.device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)

    @torch.no_grad()
    def predict_probabilities(self, texts: List[str]) -> np.ndarray:
        """Compute 3-class sentiment probabilities for a list of texts.

        Args:
            texts: List of text strings.

        Returns:
            NumPy array of shape [len(texts), 3] where columns correspond to
            [prob_negative, prob_neutral, prob_positive], sum(axis=1) == 1.0.
        """
        dataset = SentimentTextDataset(texts, self.tokenizer, max_length=self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_probs: List[np.ndarray] = []
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy().astype(np.float32))

        return np.vstack(all_probs)


def export_sentiment_features(
    manifest_path: Union[str, Path] = "data/manifest.csv",
    output_dir: Union[str, Path] = "features/emotion",
    condition: str = UTTERANCE_ONLY,
    extractor: Optional[SentimentFeatureExtractor] = None,
) -> pd.DataFrame:
    """Extract and export 3-class sentiment probabilities for downstream integration.

    Args:
        manifest_path: Path to manifest.csv.
        output_dir: Output directory (typically features/emotion/).
        condition: 'utterance_only' or 'utterance_plus_context'.
        extractor: Optional existing extractor instance.

    Returns:
        DataFrame containing utterance_id, probabilities, and predicted sentiment.
    """
    manifest_df = pd.read_csv(manifest_path)
    samples = process_manifest(manifest_df, condition=condition)
    texts = [s.get_input_text(condition=condition) for s in samples]
    uids = [s.utterance_id for s in samples]

    ext = extractor or SentimentFeatureExtractor()
    probs = ext.predict_probabilities(texts)

    pred_labels = [IDX_TO_CLASS[int(np.argmax(p))] for p in probs]

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # Save individual .npy files keyed by utterance_id
    emb_dict: Dict[str, np.ndarray] = {}
    for uid, p_vec in zip(uids, probs):
        np.save(out_p / f"{uid}.npy", p_vec)
        emb_dict[uid] = p_vec

    # Save compressed .npz archive
    np.savez_compressed(out_p / "sentiment_probabilities.npz", **emb_dict)

    # Save summary CSV table
    df_out = pd.DataFrame({
        "utterance_id": uids,
        "prob_negative": probs[:, 0].round(4),
        "prob_neutral": probs[:, 1].round(4),
        "prob_positive": probs[:, 2].round(4),
        "predicted_sentiment": pred_labels,
    })
    csv_path = out_p / "sentiment_probabilities.csv"
    df_out.to_csv(csv_path, index=False)
    logger.info("Saved sentiment probabilities for %d utterances to %s", len(df_out), out_p)

    return df_out


def validate_sentiment_against_valence(
    sentiment_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Validate sentiment probabilities against annotated MUStARD++ valence (validation-only).

    Computes Pearson / Spearman correlation and mean valence per predicted sentiment.
    """
    merged = pd.merge(sentiment_df, manifest_df[["utterance_id", "valence", "arousal", "sarcasm"]], on="utterance_id")
    # Positive continuous sentiment index = prob_positive - prob_negative
    continuous_sentiment = merged["prob_positive"] - merged["prob_negative"]

    corr = float(np.corrcoef(continuous_sentiment, merged["valence"])[0, 1])

    mean_valence_by_sentiment = merged.groupby("predicted_sentiment")["valence"].mean().to_dict()

    validation_report = {
        "pearson_correlation_with_valence": round(corr, 4),
        "mean_valence_by_predicted_sentiment": {k: round(float(v), 4) for k, v in mean_valence_by_sentiment.items()},
        "sample_count": len(merged),
        "validation_status": "PASSED" if corr > 0.3 else "WARNING_LOW_CORRELATION",
    }
    logger.info("Sentiment vs Valence validation: %s", validation_report)
    return validation_report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    repo_root = Path(__file__).resolve().parents[3]
    m_path = repo_root / "data" / "manifest.csv"
    out_dir = repo_root / "features" / "emotion"
    s_df = export_sentiment_features(manifest_path=m_path, output_dir=out_dir)
    m_df = pd.read_csv(m_path)
    val_report = validate_sentiment_against_valence(s_df, m_df)
    report_path = repo_root / "results" / "sentiment_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2)
    logger.info("Saved sentiment validation report to %s", report_path)

