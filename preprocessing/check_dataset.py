import os

DATASET_PATH = "dataset/RAVDESS"

total_files = 0

for actor_folder in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor_folder)

    if os.path.isdir(actor_path):

        for file in os.listdir(actor_path):

            if file.endswith(".wav"):
                total_files += 1

print("Total audio files:", total_files)