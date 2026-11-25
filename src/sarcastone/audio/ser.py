import os
import pandas as pd
import opensmile

AUDIO_DIR = "data/processed/audio"
OUTPUT_DIR = "features/emotion"

os.makedirs(OUTPUT_DIR, exist_ok=True)

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

audio_files = [
    f for f in os.listdir(AUDIO_DIR)
    if f.lower().endswith(".wav")
]

print("Total audio files:", len(audio_files))

results = []

for index, filename in enumerate(audio_files):

    filepath = os.path.join(AUDIO_DIR, filename)

    try:
        features = smile.process_file(filepath)

        features.insert(
            0,
            "KEY",
            filename.replace(".wav", "")
        )

        results.append(features.reset_index(drop=True))

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(audio_files)}")

    except Exception as e:
        print("Error:", filename, e)

if results:
    df = pd.concat(results, ignore_index=True)

    output_path = os.path.join(
        OUTPUT_DIR,
        "opensmile_emotion_features.csv"
    )

    df.to_csv(output_path, index=False)

    print("\nExtraction completed!")
    print("Successfully processed:", len(df))
    print("Feature count:", len(df.columns) - 1)
    print("Saved to:", output_path)
else:
    print("No features extracted.")