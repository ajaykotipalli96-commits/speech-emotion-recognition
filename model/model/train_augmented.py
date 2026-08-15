import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ==========================================
# Load augmented features
# ==========================================

data = pd.read_csv(
    "features/augmented_features.csv"
)

print("Augmented dataset loaded!")
print("Dataset shape:", data.shape)


# ==========================================
# Features and labels
# ==========================================

X = data.drop(
    columns=["audio_path", "emotion"]
)

y = data["emotion"]


# ==========================================
# Label encoder
# ==========================================

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


# ==========================================
# Train / Test split
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
# SVM Model
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
            kernel="rbf"
        )
    )

])


# ==========================================
# Train
# ==========================================

print("\nTraining augmented SVM...")

model.fit(
    X_train,
    y_train
)

print("Augmented SVM training completed!")


# ==========================================
# Training accuracy
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
# Test accuracy
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
    "model/augmented_emotion_model.pkl"
)

joblib.dump(
    label_encoder,
    "model/augmented_label_encoder.pkl"
)


print("\nAugmented model saved successfully!")