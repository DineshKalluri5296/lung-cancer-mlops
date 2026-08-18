import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Read training data
train = pd.read_csv("artifacts/train.csv")

print("Training data shape:", train.shape)

# Separate features and target
X = train.drop("LUNG_CANCER", axis=1)
y = train["LUNG_CANCER"]

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X, y)

# Check training accuracy
predictions = model.predict(X)

accuracy = accuracy_score(y, predictions)

print("Training accuracy:", accuracy)

# Save model
joblib.dump(
    model,
    "model/lung_cancer_model.pkl"
)

print("Model saved successfully!")