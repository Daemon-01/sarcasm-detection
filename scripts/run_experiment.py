"""CLI entry point to run SarcasTone text branch experiments.

Reproduces:
- TF-IDF + Logistic Regression Baseline
- RoBERTa-base (Frozen) + SarcasmClassifierHead
- Pre-trained 3-class Sentiment Model Validation
- Multi-seed × 5-fold cross-validation on frozen folds
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to sys.path if not present
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from sarcastone.eval.stats import compute_paired_comparison, format_comparison_table
from sarcastone.text.embed import run_roberta_experiment
from sarcastone.text.sentiment import (
    export_sentiment_features,
    validate_sentiment_against_valence,
)
from sarcastone.text.tfidf import run_tfidf_baseline_experiment
from sarcastone.utils.seeds import EXPERIMENT_SEEDS

logger = logging.getLogger("sarcastone.cli")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SarcasTone Text Experiments")
    parser.add_argument(
        "--model",
        choices=["all", "tfidf", "roberta", "sentiment"],
        default="all",
        help="Model experiment to run (default: all)",
    )
    parser.add_argument(
        "--manifest",
        default="data/manifest.csv",
        help="Path to manifest.csv",
    )
    parser.add_argument(
        "--folds",
        default="data/folds.json",
        help="Path to folds.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=EXPERIMENT_SEEDS,
        help="Seeds for experiment (default: 42 123 2026)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model in ["all", "tfidf"]:
        logger.info("=== Running TF-IDF Baseline Experiment ===")
        tfidf_report = run_tfidf_baseline_experiment(
            manifest_path=args.manifest,
            folds_path=args.folds,
            seeds=args.seeds,
            output_dir=out_dir,
        )

    if args.model in ["all", "roberta"]:
        logger.info("=== Running RoBERTa Frozen Encoder Experiment ===")
        roberta_report = run_roberta_experiment(
            manifest_path=args.manifest,
            folds_path=args.folds,
            seeds=args.seeds,
            output_dir=out_dir,
            feature_dir="features/text",
        )

    if args.model in ["all", "sentiment"]:
        logger.info("=== Running Sentiment Feature Extraction & Validation ===")
        sent_df = export_sentiment_features(manifest_path=args.manifest)
        import pandas as pd
        m_df = pd.read_csv(args.manifest)
        val_rep = validate_sentiment_against_valence(sent_df, m_df)
        with open(out_dir / "sentiment_validation_report.json", "w", encoding="utf-8") as f:
            json.dump(val_rep, f, indent=2)

    logger.info("=== Experiments finished successfully ===")


if __name__ == "__main__":
    main()

