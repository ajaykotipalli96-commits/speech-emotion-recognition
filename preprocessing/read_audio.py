import librosa

audio_path =  "C:\Users\AJAY\Documents\Audacity\audio.wav"

audio, sample_rate = librosa.load(audio_path, sr=None)

print("Audio loaded successfully!")
print("Sample rate:", sample_rate)
print("Audio duration:", len(audio) / sample_rate, "seconds")