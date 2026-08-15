import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, classification_report


# =====================================================
# Load original RAVDESS features
# =====================================================

data = pd.read_csv(
    "features/improved_features.csv"
)

print("Original dataset loaded!")
print("Dataset shape:", data.shape)


# =====================================================
# Load augmented model
# =====================================================

model = joblib.load(
    "model/augmented_emotion_model.pkl"
)

label_encoder = joblib.load(
    "model/augmented_label_encoder.pkl"
)


# =====================================================
# Get actor number
# =====================================================

def get_actor(path):

    path = str(path).replace("\\", "/")

    parts = path.split("/")

    for part in parts:

        if part.lower().startswith("actor_"):

            return int(
                part.split("_")[1]
            )

    return None


# =====================================================
# Identify actors
# =====================================================

data["actor"] = data["audio_path"].apply(
    get_actor
)


# =====================================================
# Completely unseen test actors
# =====================================================

test_actors = [
    20, 21, 22, 23, 24
]


test_data = data[
    data["actor"].isin(test_actors)
].copy()


print("\nTest actors:", test_actors)
print(
    "Test samples:",
    len(test_data)
)


# =====================================================
# Prepare test features
# =====================================================

X_test = test_data.drop(
    columns=[
        "audio_path",
        "emotion",
        "actor"
    ]
)

y_test = label_encoder.transform(
    test_data["emotion"]
)


# =====================================================
# IMPORTANT:
# Convert DataFrame to NumPy
# This avoids feature-name mismatch.
# =====================================================

X_test = X_test.to_numpy()


print(
    "Test feature count:",
    X_test.shape[1]
)


# =====================================================
# Prediction
# =====================================================

print("\nTesting augmented model...")

y_pred = model.predict(
    X_test
)


# =====================================================
# Accuracy
# =====================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n====================================")
print("ACTOR-INDEPENDENT TEST RESULT")
print("====================================")

print(
    "Test Accuracy:",
    accuracy
)

print(
    "Test Accuracy (%):",
    accuracy * 100
)


# =====================================================
# Classification report
# =====================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)