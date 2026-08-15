import os
import librosa
import numpy as np
import pandas as pd

DATASET_PATH = "dataset/RAVDESS"

emotion_map = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fear",
    7: "disgust",
    8: "surprise"
}

data = []

for actor_folder in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor_folder)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(actor_path, file)

        parts = file.replace(".wav", "").split("-")
        emotion = emotion_map[int(parts[2])]

        audio, sr = librosa.load(
            file_path,
            sr=16000
        )

        audio = librosa.util.normalize(audio)

        target_length = 3 * sr

        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            audio = np.pad(
                audio,
                (0, target_length - len(audio))
            )

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )

        # MFCC Delta
        mfcc_delta = librosa.feature.delta(mfcc)

        # Chroma
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=sr
        )

        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=64
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        # Mean + Standard Deviation
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),

            np.mean(mfcc_delta, axis=1),
            np.std(mfcc_delta, axis=1),

            np.mean(chroma, axis=1),
            np.std(chroma, axis=1),

            np.mean(mel_db, axis=1),
            np.std(mel_db, axis=1)
        ])

        data.append(
            [file_path, emotion, *features]
        )

        print("Processed:", file)

# Create feature names
columns = ["audio_path", "emotion"]

for i in range(40):
    columns.append(f"mfcc_mean_{i+1}")

for i in range(40):
    columns.append(f"mfcc_std_{i+1}")

for i in range(40):
    columns.append(f"delta_mean_{i+1}")

for i in range(40):
    columns.append(f"delta_std_{i+1}")

for i in range(12):
    columns.append(f"chroma_mean_{i+1}")

for i in range(12):
    columns.append(f"chroma_std_{i+1}")

for i in range(64):
    columns.append(f"mel_mean_{i+1}")

for i in range(64):
    columns.append(f"mel_std_{i+1}")

df = pd.DataFrame(data, columns=columns)

df.to_csv(
    "features/improved_features.csv",
    index=False
)

print("\nImproved feature extraction completed!")
print("Total audio files:", len(df))
print("Total features:", len(columns) - 2)
print("Saved to: features/improved_features.csv")