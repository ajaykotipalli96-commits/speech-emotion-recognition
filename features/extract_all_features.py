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

features_data = []

for actor_folder in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor_folder)

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(actor_path, file)

        # Get emotion from filename
        parts = file.replace(".wav", "").split("-")
        emotion_code = int(parts[2])
        emotion = emotion_map[emotion_code]

        # Load audio
        audio, sample_rate = librosa.load(
            file_path,
            sr=16000
        )

        # Normalize
        audio = librosa.util.normalize(audio)

        # Fixed length
        target_length = 3 * sample_rate

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
            sr=sample_rate,
            n_mfcc=40
        )

        # Chroma
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=sample_rate
        )

        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=128
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        # Convert each feature matrix to fixed-size values
        mfcc_mean = np.mean(mfcc, axis=1)
        chroma_mean = np.mean(chroma, axis=1)
        mel_mean = np.mean(mel_db, axis=1)

        combined_features = np.concatenate(
            [mfcc_mean, chroma_mean, mel_mean]
        )

        features_data.append(
            [file_path, emotion, *combined_features]
        )

        print("Processed:", file)

# Create column names
feature_columns = []

for i in range(40):
    feature_columns.append(f"mfcc_{i+1}")

for i in range(12):
    feature_columns.append(f"chroma_{i+1}")

for i in range(128):
    feature_columns.append(f"mel_{i+1}")

columns = ["audio_path", "emotion"] + feature_columns

df = pd.DataFrame(
    features_data,
    columns=columns
)

df.to_csv(
    "features/features.csv",
    index=False
)

print("\nFeature extraction completed!")
print("Total audio files processed:", len(df))
print("Total features per audio:", len(feature_columns))
print("Saved to: features/features.csv")