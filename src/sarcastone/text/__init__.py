"""Text processing package for SarcasTone.

Includes deterministic text normalization, context serialization, TF-IDF representations,
RoBERTa embeddings, and sentiment extraction.
"""

from sarcastone.text.preprocess import (
    TURN_DELIMITER,
    UTTERANCE_ONLY,
    UTTERANCE_PLUS_CONTEXT,
    VALID_CONDITIONS,
    TextSample,
    deserialize_context,
    get_text_inputs_and_labels,
    normalize_whitespace,
    prepare_context_turn,
    prepare_target_text,
    process_manifest,
    process_sample,
    serialize_context,
)
from sarcastone.text.embed import (
    RobertaFeatureExtractor,
    SarcasmClassifierHead,
    export_roberta_embeddings,
    mean_pooling,
    run_roberta_cv_single_seed,
    run_roberta_experiment,
)
from sarcastone.text.sentiment import (
    SentimentFeatureExtractor,
    export_sentiment_features,
    validate_sentiment_against_valence,
)
from sarcastone.text.tfidf import (
    build_tfidf_vectorizer,
    run_tfidf_cv_single_seed,
    run_tfidf_experiment,
    train_eval_tfidf_fold,
)

__all__ = [
    "TURN_DELIMITER",
    "UTTERANCE_ONLY",
    "UTTERANCE_PLUS_CONTEXT",
    "VALID_CONDITIONS",
    "TextSample",
    "normalize_whitespace",
    "prepare_target_text",
    "prepare_context_turn",
    "serialize_context",
    "deserialize_context",
    "process_sample",
    "process_manifest",
    "get_text_inputs_and_labels",
    "build_tfidf_vectorizer",
    "train_eval_tfidf_fold",
    "run_tfidf_cv_single_seed",
    "run_tfidf_experiment",
    "mean_pooling",
    "SarcasmClassifierHead",
    "RobertaFeatureExtractor",
    "export_roberta_embeddings",
    "run_roberta_cv_single_seed",
    "run_roberta_experiment",
    "SentimentFeatureExtractor",
    "export_sentiment_features",
    "validate_sentiment_against_valence",
]

