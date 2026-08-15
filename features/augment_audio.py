import os
import librosa
import soundfile as sf
import numpy as np


# Original dataset
DATASET_PATH = "dataset/RAVDESS"

# New augmented dataset
AUGMENTED_PATH = "dataset/AUGMENTED_RAVDESS"

os.makedirs(AUGMENTED_PATH, exist_ok=True)


# Add a small amount of noise
def add_noise(audio):
    noise = np.random.normal(
        0,
        0.005,
        len(audio)
    )

    return audio + noise


# Slight pitch change
def change_pitch(audio, sr):
    return librosa.effects.pitch_shift(
        audio,
        sr=sr,
        n_steps=1
    )


# Slight time stretching
def time_stretch(audio):
    return librosa.effects.time_stretch(
        audio,
        rate=1.05
    )


processed = 0


for actor_folder in os.listdir(DATASET_PATH):

    actor_path = os.path.join(
        DATASET_PATH,
        actor_folder
    )

    if not os.path.isdir(actor_path):
        continue

    # Create actor folder
    output_actor_path = os.path.join(
        AUGMENTED_PATH,
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

        # Load audio
        audio, sr = librosa.load(
            input_path,
            sr=16000
        )


        # --------------------------------
        # 1. Original audio
        # --------------------------------

        original_path = os.path.join(
            output_actor_path,
            file
        )

        sf.write(
            original_path,
            audio,
            sr
        )


        # --------------------------------
        # 2. Noise augmentation
        # --------------------------------

        noisy_audio = add_noise(audio)

        noisy_name = file.replace(
            ".wav",
            "_noise.wav"
        )

        sf.write(
            os.path.join(
                output_actor_path,
                noisy_name
            ),
            noisy_audio,
            sr
        )


        # --------------------------------
        # 3. Pitch augmentation
        # --------------------------------

        pitched_audio = change_pitch(
            audio,
            sr
        )

        pitched_name = file.replace(
            ".wav",
            "_pitch.wav"
        )

        sf.write(
            os.path.join(
                output_actor_path,
                pitched_name
            ),
            pitched_audio,
            sr
        )


        # --------------------------------
        # 4. Time augmentation
        # --------------------------------

        stretched_audio = time_stretch(
            audio
        )

        stretched_name = file.replace(
            ".wav",
            "_stretch.wav"
        )

        sf.write(
            os.path.join(
                output_actor_path,
                stretched_name
            ),
            stretched_audio,
            sr
        )


        processed += 1

        print("Processed:", file)


print("\nAudio augmentation completed!")
print("Original files processed:", processed)
print(
    "Augmented dataset saved to:",
    AUGMENTED_PATH
)