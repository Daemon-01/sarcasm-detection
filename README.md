# SarcasTone: Multimodal Sarcasm Detection Using Text and Speech Prosody

A reproducible NLP research project for sarcasm detection on the [MUStARD++](https://github.com/cfiltnlp/MUStARD) dataset.
Current scope: **Text branch only** (speaker-independent 5-fold cross-validation).

---

## Overview

| Component | Status | Accuracy (CV Mean ± Std) | Macro F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression (Utterance Only) | ✅ Complete | 54.32% ± 0.00% | 53.98% |
| TF-IDF + Logistic Regression (Utterance + Context) | ✅ Complete | 56.59% ± 0.00% | 56.23% |
| RoBERTa-base Frozen (Utterance Only) | ✅ Complete | **59.80% ± 0.44%** | **59.61%** |
| RoBERTa-base Frozen (Utterance + Context) | ✅ Complete | 58.87% ± 0.35% | 58.50% |
| Sentiment feature extraction & validation | ✅ Complete | — | — |
| Speech branch | 🔲 Not in scope | — | — |
| Multimodal fusion | 🔲 Not in scope | — | — |

**Evaluation protocol:** 5 speaker-independent folds (`data/folds.json`), seeds [42, 123, 2026].

---

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Set Python path
export PYTHONPATH=src          # Linux/macOS
$env:PYTHONPATH="src"         # PowerShell
```

---

## Project Structure

```
SarcasmDetection(nlp)/
├── data/
│   ├── mustard++_text.csv          # Raw MUStARD++ dataset
│   ├── manifest.csv                # 1,202 target samples
│   ├── manifest_utterances.csv     # 6,041 all turns
│   └── folds.json                  # Frozen 5-fold speaker-independent splits
├── features/
│   ├── text/                       # 768-dim RoBERTa embeddings (.npy, .npz)
│   └── emotion/                    # 3-class sentiment probabilities (.npy, .npz, .csv)
├── results/
│   ├── tfidf_baseline_results.json
│   ├── roberta_experiment_results.json
│   ├── sentiment_validation_report.json
│   └── model_comparison_report.json
├── src/sarcastone/
│   ├── data/
│   │   ├── manifest.py             # Dataset loading and manifest generation
│   │   └── folds.py                # Speaker-independent 5-fold CV splits
│   ├── text/
│   │   ├── preprocess.py           # Text normalization, context serialization
│   │   ├── tfidf.py                # TF-IDF + LogReg baseline
│   │   ├── embed.py                # Frozen RoBERTa + classifier head
│   │   └── sentiment.py            # Pre-trained 3-class sentiment model
│   ├── eval/
│   │   ├── metrics.py              # Accuracy, F1, ROC-AUC, etc.
│   │   ├── cv_runner.py            # Multi-seed cross-validation engine
│   │   ├── subsets.py              # Sentiment-incongruent subset evaluator
│   │   └── stats.py                # Paired statistical tests
│   └── utils/
│       ├── seeds.py                # Experiment seeds [42, 123, 2026]
│       ├── io.py                   # JSON/NumPy I/O helpers
│       └── logging.py              # Logging setup
├── scripts/
│   ├── run_experiment.py           # Reproduce all experiments (CLI)
│   └── extract_text.py             # Extract text/sentiment features (CLI)
├── tests/                          # 39 unit tests (100% passing)
└── docs/sarcastone_text_branch_okf/ # OKF knowledge bundle (source of truth)
```

---

## How to Reproduce

### Run all experiments
```bash
# Full experiment suite (TF-IDF + RoBERTa + Sentiment)
.venv\Scripts\python scripts/run_experiment.py --model all

# Individual models
.venv\Scripts\python scripts/run_experiment.py --model tfidf
.venv\Scripts\python scripts/run_experiment.py --model roberta
.venv\Scripts\python scripts/run_experiment.py --model sentiment
```

### Extract features only
```bash
.venv\Scripts\python scripts/extract_text.py --condition both
```

### Run tests
```bash
.venv\Scripts\python -m unittest discover -s tests -p "test_*.py"
```

---

## Incongruent-Subset Definition

A sample is **sentiment-incongruent** when the literal text sentiment conflicts with the communicative intent:

- **Positive surface sentiment + sarcasm=1** (e.g., "Oh *brilliant*" sarcastically): 127 samples.
- **Negative surface sentiment + sarcasm=0** (literal complaint, non-sarcastic): 184 samples.

Total incongruent: **311 of 1,202 samples** (25.9%).

RoBERTa (seed 42) achieves **62.7%** accuracy on incongruent samples vs. **58.7%** on congruent samples — indicating the model learns surface-incongruence cues even without speech prosody.

---

## Key Findings

| Finding | Value |
|---|---|
| RoBERTa vs TF-IDF accuracy gain | +5.48% (p = 1.0×10⁻⁵, paired t-test) |
| RoBERTa vs TF-IDF Macro F1 gain | +5.63% (p = 1.2×10⁻⁵) |
| Context gain (TF-IDF) | +2.27% accuracy |
| Context gain (RoBERTa frozen) | −0.93% (mean pooling dilutes target cues) |
| Sentiment vs valence Pearson r | +0.22 (validation-only) |

---

## OKF Knowledge Bundle

Design decisions, evaluation conventions, and data-pipeline specifications are documented in:

```
docs/sarcastone_text_branch_okf/index.md
```

See `src/sarcastone/agents.md` for agent instruction rules.
