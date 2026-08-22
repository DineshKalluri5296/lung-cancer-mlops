import pandas as pd
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


TRAIN_FILE = "artifacts/train.csv"
MODEL_FILE = "model/model.pkl"

TARGET = "LUNG_CANCER"


# Create model folder
os.makedirs("model", exist_ok=True)


# Load training data
train = pd.read_csv(TRAIN_FILE)

print("Training data shape:", train.shape)


# Separate X and y
X_train = train.drop(columns=[TARGET])
y_train = train[TARGET]


# Create model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# Train
model.fit(
    X_train,
    y_train
)


# Training accuracy
predictions = model.predict(X_train)

accuracy = accuracy_score(
    y_train,
    predictions
)

print("Training accuracy:", accuracy)


# Save model
joblib.dump(
    model,
    MODEL_FILE
)

print("Model saved to:", MODEL_FILE)