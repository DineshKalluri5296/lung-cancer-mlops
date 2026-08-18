import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load test data
test = pd.read_csv("artifacts/test.csv")

# Load model
model = joblib.load("model/lung_cancer_model.pkl")

# Split features and target
X_test = test.drop("LUNG_CANCER", axis=1)
y_test = test["LUNG_CANCER"]

# Predict
predictions = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)