import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# Load improved features
data = pd.read_csv("features/improved_features.csv")

print("Improved dataset loaded!")
print("Dataset shape:", data.shape)


# Separate features and emotion
X = data.drop(columns=["audio_path", "emotion"])
y = data["emotion"]


# Encode emotion labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# Create SVM pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=10,
        gamma="scale"
    ))
])


# Train model
model.fit(X_train, y_train)

print("Improved SVM model training completed!")


# Training accuracy
train_predictions = model.predict(X_train)

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

print("Training accuracy:", train_accuracy)


# Save model
joblib.dump(
    model,
    "model/improved_emotion_model.pkl"
)

joblib.dump(
    label_encoder,
    "model/improved_label_encoder.pkl"
)

print("Improved model saved successfully!")