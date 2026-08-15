import os
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

        parts = file.replace(".wav", "").split("-")

        emotion_code = int(parts[2])

        emotion = emotion_map[emotion_code]

        audio_path = os.path.join(actor_path, file)

        data.append([audio_path, emotion])

df = pd.DataFrame(data, columns=["audio_path", "emotion"])

df.to_csv("dataset/audio_labels.csv", index=False)

print("Labels created successfully!")
print("Total audio files:", len(df))

print("\nEmotion distribution:")
print(df["emotion"].value_counts())