import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

audio_path = ROOT / "features/prosody/prosody_features.csv"
labels_path = ROOT / "data/processed/mustard++_text_clean.csv"

# Read audio keys
with open(audio_path, encoding="utf-8") as file:
    audio_keys = [row["KEY"] for row in csv.DictReader(file)]

# Read target utterance labels
label_dict = {}

with open(labels_path, encoding="utf-8") as file:
    for row in csv.DictReader(file):
        key = row["KEY"]

        if key.endswith("_u") and row["Sarcasm"].strip():
            label_dict[key] = int(float(row["Sarcasm"]))

# Match audio and labels
matched = [
    (key, label_dict[key])
    for key in audio_keys
    if key in label_dict
]

print("Audio samples:", len(audio_keys))
print("Target labels:", len(label_dict))
print("Matched samples:", len(matched))
print("Unmatched samples:", len(audio_keys) - len(matched))

print("Non-sarcastic:", sum(label == 0 for _, label in matched))
print("Sarcastic:", sum(label == 1 for _, label in matched))