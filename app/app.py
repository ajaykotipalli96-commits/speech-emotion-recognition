import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import os


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="centered"
)


# =========================================================
# TITLE
# =========================================================

st.title("🎙️ Speech Emotion Recognition")

st.write(
    "Upload a WAV audio file and the system will predict "
    "the emotion expressed in the speech."
)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("model/clean_emotion_model.pkl")

label_encoder = joblib.load(
    "model/clean_label_encoder.pkl"
)


# =========================================================
# AUDIO UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a WAV audio file",
    type=["wav"]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    st.audio(
        uploaded_file,
        format="audio/wav"
    )

    if st.button("Predict Emotion"):

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp_file:

            temp_file.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_file.name


        try:

            # Load audio
            audio, sample_rate = librosa.load(
                temp_path,
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


            # -------------------------
            # MFCC
            # -------------------------

            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sample_rate,
                n_mfcc=40
            )


            # -------------------------
            # MFCC Delta
            # -------------------------

            mfcc_delta = librosa.feature.delta(
                mfcc
            )


            # -------------------------
            # Chroma
            # -------------------------

            chroma = librosa.feature.chroma_stft(
                y=audio,
                sr=sample_rate
            )


            # -------------------------
            # Mel Spectrogram
            # -------------------------

            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_mels=64
            )

            mel_db = librosa.power_to_db(
                mel,
                ref=np.max
            )


            # -------------------------
            # Create 312 features
            # -------------------------

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


            # Reshape
            features = features.reshape(
                1, -1
            )


            # -------------------------
            # Prediction
            # -------------------------

            prediction = model.predict(
                features
            )

            emotion = label_encoder.inverse_transform(
                prediction
            )[0]


            # -------------------------
            # Display result
            # -------------------------

            st.success(
                f"Predicted Emotion: {emotion.upper()}"
            )


        finally:

            # Delete temporary file
            if os.path.exists(temp_path):

                os.remove(temp_path)