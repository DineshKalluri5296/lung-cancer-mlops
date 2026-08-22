import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


TEST_FILE = "artifacts/test.csv"
MODEL_FILE = "model/model.pkl"

TARGET = "LUNG_CANCER"


# Load test data
test = pd.read_csv(TEST_FILE)


# Load model
model = joblib.load(MODEL_FILE)


# Separate X and y
X_test = test.drop(columns=[TARGET])
y_test = test[TARGET]


# Predict
predictions = model.predict(X_test)


# Metrics
accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)


# Print results
print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)