import librosa
import numpy as np
import joblib


# =========================================================
# 1. AUDIO FILE
# =========================================================

audio_path = r"C:\Users\AJAY\Downloads\prettyjohn1-sad-background-music_29sec-489884.wav"


# =========================================================
# 2. LOAD BEST TRAINED MODEL
# =========================================================

model = joblib.load("model/best_emotion_model.pkl")
label_encoder = joblib.load("model/best_label_encoder.pkl")


# =========================================================
# 3. LOAD AUDIO
# =========================================================

audio, sample_rate = librosa.load(
    audio_path,
    sr=16000
)

print("Audio loaded successfully!")
print("Sample rate:", sample_rate)
print("Original duration:", len(audio) / sample_rate, "seconds")


# =========================================================
# 4. NORMALIZE AUDIO
# =========================================================

audio = librosa.util.normalize(audio)


# =========================================================
# 5. FIX AUDIO LENGTH TO 3 SECONDS
# =========================================================

target_length = 3 * sample_rate

if len(audio) > target_length:

    audio = audio[:target_length]

elif len(audio) < target_length:

    audio = np.pad(
        audio,
        (0, target_length - len(audio))
    )


print("Final duration:", len(audio) / sample_rate, "seconds")


# =========================================================
# 6. MFCC FEATURES
# =========================================================

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=40
)


# =========================================================
# 7. MFCC DELTA FEATURES
# =========================================================

mfcc_delta = librosa.feature.delta(mfcc)


# =========================================================
# 8. CHROMA FEATURES
# =========================================================

chroma = librosa.feature.chroma_stft(
    y=audio,
    sr=sample_rate
)


# =========================================================
# 9. MEL SPECTROGRAM
# =========================================================

mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sample_rate,
    n_mels=64
)

mel_db = librosa.power_to_db(
    mel,
    ref=np.max
)


# =========================================================
# 10. CREATE 312 FEATURES
# =========================================================

features = np.concatenate([

    # MFCC mean
    np.mean(mfcc, axis=1),

    # MFCC standard deviation
    np.std(mfcc, axis=1),

    # MFCC Delta mean
    np.mean(mfcc_delta, axis=1),

    # MFCC Delta standard deviation
    np.std(mfcc_delta, axis=1),

    # Chroma mean
    np.mean(chroma, axis=1),

    # Chroma standard deviation
    np.std(chroma, axis=1),

    # Mel mean
    np.mean(mel_db, axis=1),

    # Mel standard deviation
    np.std(mel_db, axis=1)

])


# =========================================================
# 11. CHECK FEATURE COUNT
# =========================================================

print("Feature count:", len(features))


# =========================================================
# 12. RESHAPE FEATURES
# =========================================================

features = features.reshape(1, -1)


# =========================================================
# 13. PREDICT EMOTION
# =========================================================

prediction = model.predict(features)


# =========================================================
# 14. CONVERT NUMBER TO EMOTION NAME
# =========================================================

emotion = label_encoder.inverse_transform(
    prediction
)


# =========================================================
# 15. DISPLAY RESULT
# =========================================================

print("\n===================================")
print("     SPEECH EMOTION RECOGNITION")
print("===================================")
print("Predicted Emotion:", emotion[0])
print("===================================")