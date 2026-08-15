import librosa
import numpy as np

audio_path = "dataset/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"

# Load audio
audio, sample_rate = librosa.load(
    audio_path,
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

# -------------------------
# MFCC
# -------------------------
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=40
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
    n_mels=128
)

# Convert Mel spectrogram to decibel scale
mel_db = librosa.power_to_db(mel, ref=np.max)

print("Feature extraction completed!")

print("MFCC shape:", mfcc.shape)
print("Chroma shape:", chroma.shape)
print("Mel Spectrogram shape:", mel_db.shape)