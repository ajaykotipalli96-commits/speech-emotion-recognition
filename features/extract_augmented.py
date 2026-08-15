import os
import librosa
import numpy as np
import pandas as pd


DATASET_PATH = "dataset/AUGMENTED_RAVDESS"
OUTPUT_FILE = "features/augmented_features.csv"


features_list = []


for actor_folder in os.listdir(DATASET_PATH):

    actor_path = os.path.join(
        DATASET_PATH,
        actor_folder
    )

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(
            actor_path,
            file
        )

        try:

            # Load audio
            audio, sample_rate = librosa.load(
                file_path,
                sr=16000
            )

            # Normalize
            audio = librosa.util.normalize(audio)

            # Fixed length: 3 seconds
            target_length = 3 * sample_rate

            if len(audio) > target_length:

                audio = audio[:target_length]

            elif len(audio) < target_length:

                audio = np.pad(
                    audio,
                    (0, target_length - len(audio))
                )


            # =========================
            # MFCC
            # =========================

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=40
            )

            mfcc_delta = librosa.feature.delta(
                mfcc
            )


            # =========================
            # Chroma
            # =========================

            chroma = librosa.feature.chroma_stft(
                y=audio,
                sr=sample_rate
            )


            # =========================
            # Mel Spectrogram
            # =========================

            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_mels=64
            )

            mel_db = librosa.power_to_db(
                mel,
                ref=np.max
            )


            # =========================
            # 312 FEATURES
            # =========================

            feature_vector = np.concatenate([

                np.mean(mfcc, axis=1),
                np.std(mfcc, axis=1),

                np.mean(mfcc_delta, axis=1),
                np.std(mfcc_delta, axis=1),

                np.mean(chroma, axis=1),
                np.std(chroma, axis=1),

                np.mean(mel_db, axis=1),
                np.std(mel_db, axis=1)

            ])


            # =========================
            # Emotion from RAVDESS filename
            # =========================

            emotion_code = int(
                file.split("-")[2]
            )


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


            emotion = emotion_map.get(
                emotion_code
            )


            if emotion is None:
                continue


            # =========================
            # Save row
            # =========================

            row = list(feature_vector)

            row.append(file_path)
            row.append(emotion)

            features_list.append(row)

            print("Processed:", file)


        except Exception as e:

            print(
                "Error processing:",
                file,
                e
            )


# =========================
# Create column names
# =========================

feature_columns = []

for i in range(312):

    feature_columns.append(
        f"feature_{i+1}"
    )


feature_columns.extend([
    "audio_path",
    "emotion"
])


# =========================
# Create DataFrame
# =========================

df = pd.DataFrame(
    features_list,
    columns=feature_columns
)


# =========================
# Save
# =========================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n================================")
print("Augmented feature extraction completed!")
print("Dataset shape:", df.shape)
print("Total features:", len(feature_columns) - 2)
print("Saved to:", OUTPUT_FILE)
print("================================")