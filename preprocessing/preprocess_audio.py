import librosa
import numpy as np

audio_path = "dataset/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav"

# Load audio at 16 kHz
audio, sample_rate = librosa.load(
    audio_path,
    sr=16000
)

print("Audio loaded successfully!")
print("Sample rate:", sample_rate)
print("Original duration:", len(audio) / sample_rate, "seconds")

# Normalize audio
audio = librosa.util.normalize(audio)

print("Audio normalization completed!")
print("Maximum amplitude:", np.max(np.abs(audio)))

TARGET_DURATION = 3
TARGET_LENGTH = TARGET_DURATION * sample_rate

if len(audio) > TARGET_LENGTH:
    audio = audio[:TARGET_LENGTH]

elif len(audio) < TARGET_LENGTH:
    audio = np.pad(
        audio,
        (0, TARGET_LENGTH - len(audio))
    )

print("Fixed-length processing completed!")
print("Final duration:", len(audio) / sample_rate, "seconds")