import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Load improved features
data = pd.read_csv("features/improved_features.csv")

X = data.drop(columns=["audio_path", "emotion"])
y = data["emotion"]

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Keep final test set untouched
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# SVM pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])

# Parameters to test
parameters = {
    "svm__C": [1, 10, 30, 100],
    "svm__gamma": ["scale", 0.001, 0.01, 0.1],
    "svm__kernel": ["rbf"]
}

# Search using only training data
grid = GridSearchCV(
    pipeline,
    parameters,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

print("Starting SVM hyperparameter tuning...")

grid.fit(X_train, y_train)

print("\nTuning completed!")

print("Best parameters:")
print(grid.best_params_)

print("\nBest cross-validation accuracy:")
print(grid.best_score_)

# Evaluate best model ONCE on final test set
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)

print("\nFinal test accuracy:")
print(test_accuracy)

print("Final test accuracy (%):", test_accuracy * 100)

# Save best model
joblib.dump(
    best_model,
    "model/best_emotion_model.pkl"
)

joblib.dump(
    label_encoder,
    "model/best_label_encoder.pkl"
)

print("\nBest model saved successfully!")