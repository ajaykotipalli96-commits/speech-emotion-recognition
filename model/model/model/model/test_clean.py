import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report


# Load original RAVDESS features
data = pd.read_csv(
    "features/improved_features.csv"
)

print("Original dataset loaded!")
print("Dataset shape:", data.shape)


# Load clean model
model = joblib.load(
    "model/clean_emotion_model.pkl"
)

label_encoder = joblib.load(
    "model/clean_label_encoder.pkl"
)


# Get actor number
def get_actor(path):
    path = str(path).replace("\\", "/")

    for part in path.split("/"):
        if part.lower().startswith("actor_"):
            return int(part.split("_")[1])

    return None


data["actor"] = data["audio_path"].apply(get_actor)


# Completely unseen actors
test_actors = [20, 21, 22, 23, 24]

test_data = data[
    data["actor"].isin(test_actors)
].copy()


print("\nTest actors:", test_actors)
print("Test samples:", len(test_data))


# Prepare features
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


# Convert to NumPy to avoid feature-name mismatch
X_test = X_test.to_numpy()

print("Test feature count:", X_test.shape[1])


# Predict
print("\nTesting clean model...")

y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n====================================")
print("FINAL CLEAN ACTOR-INDEPENDENT TEST")
print("====================================")

print(
    "Test Accuracy:",
    accuracy
)

print(
    "Test Accuracy (%):",
    accuracy * 100
)


# Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)