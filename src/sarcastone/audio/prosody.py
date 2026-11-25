
import os
import numpy as np
import pandas as pd
import librosa

AUDIO_DIR = "data/processed/audio"
OUTPUT_DIR = "features/prosody"

os.makedirs(OUTPUT_DIR, exist_ok=True)

results = []

audio_files = [
    f for f in os.listdir(AUDIO_DIR)
    if f.endswith(".wav")
]

print("Total audio files:", len(audio_files))

for index, filename in enumerate(audio_files):

    filepath = os.path.join(AUDIO_DIR, filename)

    try:
        y, sr = librosa.load(filepath, sr=16000)

        duration = librosa.get_duration(y=y, sr=sr)

        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

        pitches = librosa.yin(
            y,
            fmin=65,
            fmax=400,
            sr=sr
        )

        voiced = pitches[
            (pitches >= 65) & (pitches <= 400)
        ]

        intervals = librosa.effects.split(
            y,
            top_db=25
        )

        speech_duration = sum(
            end - start for start, end in intervals
        ) / sr

        pause_duration = max(
            0,
            duration - speech_duration
        )

        results.append({
            "KEY": filename.replace(".wav", ""),
            "duration": duration,
            "pitch_mean": np.mean(voiced) if len(voiced) else 0,
            "pitch_std": np.std(voiced) if len(voiced) else 0,
            "energy_mean": np.mean(rms),
            "energy_std": np.std(rms),
            "zcr_mean": np.mean(zcr),
            "spectral_centroid_mean": np.mean(centroid),
            "speech_duration": speech_duration,
            "pause_duration": pause_duration,
            "speech_ratio": speech_duration / duration if duration else 0,
            "pause_count": max(0, len(intervals) - 1)
        })

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(audio_files)}")

    except Exception as e:
        print("Error:", filename, e)

df = pd.DataFrame(results)

output_path = os.path.join(
    OUTPUT_DIR,
    "prosody_features.csv"
)

df.to_csv(output_path, index=False)

print("\nExtraction completed!")
print("Successfully processed:", len(df))
print("Saved to:", output_path)
