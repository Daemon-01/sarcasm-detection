
import pandas as pd
import os

files = {
    "Prosody": "features/prosody/prosody_features.csv",
    "OpenSMILE": "features/emotion/opensmile_emotion_features.csv",
    "Wav2Vec2": "features/audio_ssl/wav2vec2_embeddings.csv",
    "Labels": "data/processed/mustard++_text_clean.csv"
}

for name, path in files.items():

    print("\n" + "=" * 50)
    print(name)

    if not os.path.exists(path):
        print("FILE NOT FOUND:", path)
        continue

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 3 rows:")
    print(df.head(3))
