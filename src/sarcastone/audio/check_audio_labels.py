
import pandas as pd

prosody = pd.read_csv(
    "features/prosody/prosody_features.csv"
)

opensmile = pd.read_csv(
    "features/emotion/opensmile_emotion_features.csv"
)

wav2vec = pd.read_csv(
    "features/audio_ssl/wav2vec2_embeddings.csv"
)

labels = pd.read_csv(
    "data/processed/mustard++_text_clean.csv"
)

# Get all audio utterance IDs
audio_keys = set(prosody["KEY"])

# Filter labels to only audio utterances
audio_labels = labels[labels["KEY"].isin(audio_keys)].copy()

print("Audio samples:", len(audio_keys))
print("Matched labels:", len(audio_labels))

print("\nSarcasm column values:")
print(audio_labels["Sarcasm"].value_counts(dropna=False))

# Check missing labels
print("\nMissing sarcasm labels:")
print(audio_labels["Sarcasm"].isna().sum())

# Check duplicate IDs
print("\nDuplicate label keys:")
print(audio_labels["KEY"].duplicated().sum())

# Check whether every feature file has the same IDs
print("\nKEY matching:")

print("Prosody vs OpenSMILE:",
      set(prosody["KEY"]) == set(opensmile["KEY"]))

print("Prosody vs Wav2Vec2:",
      set(prosody["KEY"]) == set(wav2vec["KEY"]))

print("Prosody vs Labels:",
      audio_keys == set(audio_labels["KEY"]))

print("\nFinal label distribution:")
print(audio_labels["Sarcasm"].value_counts())
