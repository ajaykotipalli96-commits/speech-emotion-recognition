import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load feature dataset
data = pd.read_csv("features/features.csv")

print("Dataset loaded!")
print("Dataset shape:", data.shape)

# Separate features and labels
X = data.drop(columns=["audio_path", "emotion"])
y = data["emotion"]

# Convert emotion names into numbers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

print("Model training completed!")

# Check training accuracy
train_predictions = model.predict(X_train)

train_accuracy = accuracy_score(
    y_train,
    train_predictions
)

print("Training accuracy:", train_accuracy)

import joblib

# Save trained model
joblib.dump(model, "model/emotion_model.pkl")

# Save label encoder
joblib.dump(label_encoder, "model/label_encoder.pkl")

print("Model saved successfully!")
print("Label encoder saved successfully!")