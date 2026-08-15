import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# ==========================================
# Load improved features
# ==========================================

data = pd.read_csv("features/improved_features.csv")

print("Improved dataset loaded!")
print("Dataset shape:", data.shape)


# ==========================================
# Separate features and labels
# ==========================================

X = data.drop(columns=["audio_path", "emotion"])
y = data["emotion"]


# ==========================================
# Encode labels
# ==========================================

label_encoder = joblib.load(
    "model/improved_label_encoder.pkl"
)

y_encoded = label_encoder.transform(y)


# ==========================================
# Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# Balanced SVM
# ==========================================

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),

    (
        "svm",
        SVC(
            C=10,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced"
        )
    )
])


# ==========================================
# Train
# ==========================================

print("\nTraining balanced SVM...")

model.fit(
    X_train,
    y_train
)

print("Balanced SVM training completed!")


# ==========================================
# Training Accuracy
# ==========================================

train_pred = model.predict(X_train)

train_accuracy = accuracy_score(
    y_train,
    train_pred
)

print(
    "Training accuracy:",
    train_accuracy
)


# ==========================================
# Test Accuracy
# ==========================================

test_pred = model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_pred
)

print(
    "Test accuracy:",
    test_accuracy
)

print(
    "Test accuracy (%):",
    test_accuracy * 100
)


# ==========================================
# Save model
# ==========================================

joblib.dump(
    model,
    "model/balanced_emotion_model.pkl"
)

joblib.dump(
    label_encoder,
    "model/balanced_label_encoder.pkl"
)

print("\nBalanced model saved successfully!")