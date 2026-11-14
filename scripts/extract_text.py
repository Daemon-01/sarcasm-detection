"""CLI entry point to extract text embeddings and sentiment features.

Exports:
- RoBERTa-base 768-dim embeddings (features/text/<utterance_id>.npy)
- 3-class sentiment probabilities (features/emotion/<utterance_id>.npy and sentiment_probabilities.csv)
Keyed strictly by utterance_id for downstream speech/fusion integration.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to sys.path if not present
repo_root = Path(__file__).resolve().parents[1]
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from sarcastone.text.embed import export_roberta_embeddings
from sarcastone.text.preprocess import UTTERANCE_ONLY, UTTERANCE_PLUS_CONTEXT
from sarcastone.text.sentiment import export_sentiment_features

logger = logging.getLogger("sarcastone.extract")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SarcasTone Text & Sentiment Features")
    parser.add_argument("--manifest", default="data/manifest.csv", help="Path to manifest.csv")
    parser.add_argument(
        "--condition",
        choices=[UTTERANCE_ONLY, UTTERANCE_PLUS_CONTEXT, "both"],
        default="both",
        help="Input condition for feature extraction",
    )
    parser.add_argument("--text-out", default="features/text", help="Directory for text embeddings")
    parser.add_argument("--emotion-out", default="features/emotion", help="Directory for sentiment features")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    conds = [UTTERANCE_ONLY, UTTERANCE_PLUS_CONTEXT] if args.condition == "both" else [args.condition]

    logger.info("Extracting RoBERTa text embeddings...")
    for c in conds:
        export_roberta_embeddings(manifest_path=args.manifest, output_dir=args.text_out, condition=c)

    logger.info("Extracting sentiment probabilities...")
    export_sentiment_features(manifest_path=args.manifest, output_dir=args.emotion_out)

    logger.info("Feature extraction complete.")


if __name__ == "__main__":
    main()

