
import csv
from pathlib import Path
from collections import Counter

import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


ROOT = Path(__file__).resolve().parents[3]

LABEL_FILE = ROOT / "data/processed/mustard++_text_clean.csv"

FEATURE_FILES = {
    "prosody": ROOT / "features/prosody/prosody_features.csv",
    "opensmile": ROOT / "features/emotion/opensmile_emotion_features.csv",
    "wav2vec2": ROOT / "features/audio_ssl/wav2vec2_embeddings.csv"
}


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_data(feature_names):
    label_rows = read_csv(LABEL_FILE)

    feature_data = {}

    for name, path in FEATURE_FILES.items():
        if name in feature_names:
            rows = read_csv(path)
            feature_data[name] = {
                row["KEY"]: row
                for row in rows
            }

    X = []
    y = []
    groups = []

    for row in label_rows:
        key = row["KEY"]

        # Use only target utterances, not context rows
        if not key.endswith("_u"):
            continue

        if not all(key in feature_data[name] for name in feature_names):
            continue

        features = []

        for name in feature_names:
            feature_row = feature_data[name][key]

            values = [
                float(value)
                for column, value in feature_row.items()
                if column != "KEY"
            ]

            features.extend(values)

        X.append(features)
        y.append(int(float(row["Sarcasm"])))
        groups.append(row["SPEAKER"])

    return (
        np.array(X, dtype=float),
        np.array(y, dtype=int),
        np.array(groups)
    )


def make_model(C, gamma):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=C,
            gamma=gamma
        ))
    ])


def tune_and_evaluate(name, feature_names):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    X, y, groups = load_data(feature_names)

    print("Feature shape:", X.shape)
    print("Total speakers:", len(np.unique(groups)))

    C_values = [0.1, 1, 10, 100]
    gamma_values = ["scale", 0.01, 0.1]

    outer_cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    inner_cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    fold_results = []
    selected_parameters = []

    for fold, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y, groups), start=1
    ):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        groups_train = groups[train_idx]
        groups_test = groups[test_idx]

        print(f"\nFold {fold}")
        print("Training samples:", len(train_idx))
        print("Testing samples:", len(test_idx))
        print("Training speakers:", len(np.unique(groups_train)))
        print("Testing speakers:", len(np.unique(groups_test)))

        overlap = set(groups_train) & set(groups_test)
        print("Speaker overlap:", len(overlap))

        # Prepare inner folds using only outer training data
        inner_splits = list(
            inner_cv.split(X_train, y_train, groups_train)
        )

        best_score = -1
        best_C = None
        best_gamma = None

        # Hyperparameter search
        for C in C_values:
            for gamma in gamma_values:

                inner_scores = []

                for inner_train_idx, inner_val_idx in inner_splits:

                    model = make_model(C, gamma)

                    model.fit(
                        X_train[inner_train_idx],
                        y_train[inner_train_idx]
                    )

                    predictions = model.predict(
                        X_train[inner_val_idx]
                    )

                    score = f1_score(
                        y_train[inner_val_idx],
                        predictions,
                        zero_division=0
                    )

                    inner_scores.append(score)

                mean_inner_f1 = np.mean(inner_scores)

                if mean_inner_f1 > best_score:
                    best_score = mean_inner_f1
                    best_C = C
                    best_gamma = gamma

        print("Best C:", best_C)
        print("Best gamma:", best_gamma)
        print("Best inner CV F1:", round(best_score, 4))

        selected_parameters.append((best_C, best_gamma))

        # Train best model on the complete outer training set
        best_model = make_model(best_C, best_gamma)

        best_model.fit(X_train, y_train)

        # Evaluate only on unseen outer test speakers
        predictions = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, zero_division=0
        )

        cm = confusion_matrix(y_test, predictions, labels=[0, 1])

        print("Outer test accuracy:", round(accuracy, 4))
        print("Outer test precision:", round(precision, 4))
        print("Outer test recall:", round(recall, 4))
        print("Outer test F1:", round(f1, 4))
        print("Confusion matrix:\n", cm)

        fold_results.append([
            accuracy,
            precision,
            recall,
            f1
        ])

    results = np.array(fold_results)

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]

    print("\n" + "-" * 60)
    print("FINAL NESTED CV RESULTS")
    print("-" * 60)

    for i, metric in enumerate(metric_names):
        mean = np.mean(results[:, i])
        std = np.std(results[:, i])

        print(f"{metric}: {mean:.4f} ± {std:.4f}")

    print("\nSelected parameters across outer folds:")
    for params, count in Counter(selected_parameters).items():
        print(
            f"C={params[0]}, gamma={params[1]}: "
            f"{count} fold(s)"
        )


if __name__ == "__main__":

    tune_and_evaluate(
        "OpenSMILE + SVM",
        ["opensmile"]
    )

    tune_and_evaluate(
        "Combined Audio + SVM",
        ["prosody", "opensmile", "wav2vec2"]
    )
