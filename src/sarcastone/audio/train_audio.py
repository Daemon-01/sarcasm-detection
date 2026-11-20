
import csv
from pathlib import Path
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

LABEL_PATH = ROOT / "data/processed/mustard++_text_clean.csv"

# Feature configurations

FEATURE_SETS = {
    "Prosody + SVM": [
        ROOT / "features/prosody/prosody_features.csv"
    ],

    "OpenSMILE + SVM": [
        ROOT / "features/emotion/opensmile_emotion_features.csv"
    ],

    "Wav2Vec2 + SVM": [
        ROOT / "features/audio_ssl/wav2vec2_embeddings.csv"
    ],

    "OpenSMILE + Prosody + SVM": [
        ROOT / "features/emotion/opensmile_emotion_features.csv",
        ROOT / "features/prosody/prosody_features.csv"
    ],

    "Combined Audio + SVM": [
        ROOT / "features/prosody/prosody_features.csv",
        ROOT / "features/emotion/opensmile_emotion_features.csv",
        ROOT / "features/audio_ssl/wav2vec2_embeddings.csv"
    ]
}



# Load labels and speaker IDs
labels = {}

with open(LABEL_PATH, encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        key = row["KEY"]

        if key.endswith("_u") and row["Sarcasm"].strip():
            labels[key] = {
                "label": int(float(row["Sarcasm"])),
                "speaker": row["SPEAKER"]
            }


# Function to load audio features
def load_features(feature_files):

    feature_dict = {}

    for file_path in feature_files:

        with open(file_path, encoding="utf-8") as file:
            reader = csv.DictReader(file)
            columns = [
                col for col in reader.fieldnames
                if col != "KEY"
            ]

            for row in reader:
                key = row["KEY"]

                values = [
                    float(row[col]) for col in columns
                ]

                feature_dict.setdefault(key, []).extend(values)

    # Align features, labels and speaker groups
    X = []
    y = []
    groups = []

    for key, features in feature_dict.items():

        if key in labels:
            X.append(features)
            y.append(labels[key]["label"])
            groups.append(labels[key]["speaker"])

    return (
        np.array(X, dtype=float),
        np.array(y),
        np.array(groups)
    )


# Five-fold speaker-independent cross-validation
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Store final results
all_results = {}


for model_name, feature_files in FEATURE_SETS.items():

    print("\n" + "=" * 55)
    print(model_name)
    print("=" * 55)

    X, y, groups = load_features(feature_files)

    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)
    print("Total speakers:", len(np.unique(groups)))

    fold_metrics = []
    total_confusion = np.zeros((2, 2), dtype=int)

    # Run five folds
    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups), start=1
    ):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])

        overlap = train_groups.intersection(test_groups)

        # Verify no speaker leakage
        assert len(overlap) == 0, "Speaker overlap detected!"

        # Build model
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf"))
        ])

        # Train and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test, y_pred, zero_division=0
        )

        recall = recall_score(
            y_test, y_pred, zero_division=0
        )

        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )

        cm = confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        )

        total_confusion += cm

        fold_metrics.append([
            accuracy,
            precision,
            recall,
            f1
        ])

        print(f"\nFold {fold}")
        print("Training samples:", len(train_idx))
        print("Testing samples:", len(test_idx))
        print("Training speakers:", len(train_groups))
        print("Testing speakers:", len(test_groups))
        print("Speaker overlap:", len(overlap))
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Confusion Matrix:\n", cm)

    # Calculate mean and standard deviation
    fold_metrics = np.array(fold_metrics)

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

    print("\n--- 5-Fold Average Results ---")

    for i, metric in enumerate(metric_names):

        mean_score = np.mean(fold_metrics[:, i])
        std_score = np.std(fold_metrics[:, i])

        print(
            f"{metric}: {mean_score:.4f} "
            f"+/- {std_score:.4f}"
        )

    print("\nCombined Confusion Matrix:")
    print(total_confusion)

    all_results[model_name] = {
        "metrics": fold_metrics,
        "mean": np.mean(fold_metrics, axis=0),
        "std": np.std(fold_metrics, axis=0)
    }


# Final comparison
print("\n" + "=" * 55)
print("FINAL MODEL COMPARISON")
print("=" * 55)

for model_name, results in all_results.items():

    print("\n", model_name)

    for i, metric in enumerate([
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]):

        mean_score = results["mean"][i]
        std_score = results["std"][i]

        print(
            f"{metric}: "
            f"{mean_score:.4f} +/- {std_score:.4f}"
        )
