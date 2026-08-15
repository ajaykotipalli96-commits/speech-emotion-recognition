import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load improved dataset
data = pd.read_csv("features/improved_features.csv")

# Separate features and labels
X = data.drop(columns=["audio_path", "emotion"])
y = data["emotion"]

# Load improved model and encoder
model = joblib.load("model/improved_emotion_model.pkl")
label_encoder = joblib.load("model/improved_label_encoder.pkl")

# Encode labels
y_encoded = label_encoder.transform(y)

# Same split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# Predict unseen test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Improved Model Testing Completed!")
print("Test Accuracy:", accuracy)
print("Test Accuracy (%):", accuracy * 100)

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