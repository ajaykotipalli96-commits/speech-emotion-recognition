import os
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# SETTINGS
# =========================================================

ORIGINAL_DATASET = "dataset/RAVDESS"

TRAIN_AUGMENTED = "dataset/CLEAN_AUGMENTED_RAVDESS"

TEST_ACTORS = [20, 21, 22, 23, 24]

FEATURE_FILE = "features/clean_augmented_features.csv"


# =========================================================
# EMOTION MAPPING
# =========================================================

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


# =========================================================
# CREATE TRAINING DATA
# EXCLUDE TEST ACTORS COMPLETELY
# =========================================================

os.makedirs(
    TRAIN_AUGMENTED,
    exist_ok=True
)

processed = 0

print("Creating clean training dataset...")
print("Test actors excluded:", TEST_ACTORS)


for actor_folder in os.listdir(ORIGINAL_DATASET):

    actor_path = os.path.join(
        ORIGINAL_DATASET,
        actor_folder
    )

    if not os.path.isdir(actor_path):
        continue

    actor_number = int(
        actor_folder.split("_")[1]
    )

    # Completely exclude test actors
    if actor_number in TEST_ACTORS:
        continue

    output_actor_path = os.path.join(
        TRAIN_AUGMENTED,
        actor_folder
    )

    os.makedirs(
        output_actor_path,
        exist_ok=True
    )


    for file in os.listdir(actor_path):

        if not file.endswith(".wav"):
            continue

        input_path = os.path.join(
            actor_path,
            file
        )

        audio, sr = librosa.load(
            input_path,
            sr=16000
        )


        # -----------------------------------------
        # Original
        # -----------------------------------------

        sf.write(
            os.path.join(
                output_actor_path,
                file
            ),
            audio,
            sr
        )


        # -----------------------------------------
        # Noise
        # -----------------------------------------

        noise = np.random.normal(
            0,
            0.005,
            len(audio)
        )

        noisy_audio = audio + noise

        sf.write(
            os.path.join(
                output_actor_path,
                file.replace(
                    ".wav",
                    "_noise.wav"
                )
            ),
            noisy_audio,
            sr
        )


        # -----------------------------------------
        # Pitch
        # -----------------------------------------

        pitched_audio = librosa.effects.pitch_shift(
            audio,
            sr=sr,
            n_steps=1
        )

        sf.write(
            os.path.join(
                output_actor_path,
                file.replace(
                    ".wav",
                    "_pitch.wav"
                )
            ),
            pitched_audio,
            sr
        )


        # -----------------------------------------
        # Time stretch
        # -----------------------------------------

        stretched_audio = librosa.effects.time_stretch(
            audio,
            rate=1.05
        )

        sf.write(
            os.path.join(
                output_actor_path,
                file.replace(
                    ".wav",
                    "_stretch.wav"
                )
            ),
            stretched_audio,
            sr
        )


        processed += 1

        print(
            "Processed:",
            actor_folder,
            file
        )


print("\nClean training dataset created!")
print(
    "Original training files:",
    processed
)


# =========================================================
# FEATURE EXTRACTION
# =========================================================

features = []

print("\nExtracting 312 features...")


for actor_folder in os.listdir(TRAIN_AUGMENTED):

    actor_path = os.path.join(
        TRAIN_AUGMENTED,
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


        audio, sr = librosa.load(
            file_path,
            sr=16000
        )

        audio = librosa.util.normalize(
            audio
        )


        target_length = 3 * sr


        if len(audio) > target_length:

            audio = audio[:target_length]

        elif len(audio) < target_length:

            audio = np.pad(
                audio,
                (
                    0,
                    target_length - len(audio)
                )
            )


        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )


        # MFCC Delta
        mfcc_delta = librosa.feature.delta(
            mfcc
        )


        # Chroma
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=sr
        )


        # Mel
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=64
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )


        # 312 features
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


        emotion_code = int(
            file.split("-")[2]
        )

        emotion = emotion_map[
            emotion_code
        ]


        features.append(
            [
                *feature_vector,
                file_path,
                emotion
            ]
        )


print("Feature extraction completed!")


# =========================================================
# CREATE DATAFRAME
# =========================================================

columns = [
    f"feature_{i+1}"
    for i in range(312)
]

columns += [
    "audio_path",
    "emotion"
]


df = pd.DataFrame(
    features,
    columns=columns
)


df.to_csv(
    FEATURE_FILE,
    index=False
)


print(
    "Clean feature dataset:",
    df.shape
)


# =========================================================
# TRAIN SVM
# =========================================================

X = df.drop(
    columns=[
        "audio_path",
        "emotion"
    ]
)

y = df["emotion"]


label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(
    y
)


model = Pipeline([

    (
        "scaler",
        StandardScaler()
    ),

    (
        "svm",
        SVC(
            C=10,
            gamma="scale",
            kernel="rbf"
        )
    )

])


print("\nTraining clean augmented SVM...")

model.fit(
    X,
    y_encoded
)


print(
    "Clean augmented model training completed!"
)


# =========================================================
# SAVE
# =========================================================

joblib.dump(
    model,
    "model/clean_emotion_model.pkl"
)

joblib.dump(
    label_encoder,
    "model/clean_label_encoder.pkl"
)


print(
    "\nClean model saved successfully!"
)