import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load feature dataset
data = pd.read_csv("features/features.csv")

# Separate features and labels
X = data.drop(columns=["audio_path", "emotion"])
y = data["emotion"]

# Load trained model
model = joblib.load("model/emotion_model.pkl")

# Load label encoder
label_encoder = joblib.load("model/label_encoder.pkl")

# Convert emotion labels into numbers
y_encoded = label_encoder.transform(y)

# Split data exactly like training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Testing Completed!")
print("Test Accuracy:", accuracy)

# Classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))