import pandas as pd
import joblib
import os


MODEL_FILE = "model/model.pkl"
TEST_FILE = "artifacts/test.csv"

TARGET = "LUNG_CANCER"


def main():

    print("==============================")
    print("LUNG CANCER INFERENCE")
    print("==============================")

    # Check model
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    # Check test data
    if not os.path.exists(TEST_FILE):
        raise FileNotFoundError(
            f"Test data not found: {TEST_FILE}"
        )

    # Load model
    model = joblib.load(MODEL_FILE)

    # Load test data
    test = pd.read_csv(TEST_FILE)

    # Separate features
    X_test = test.drop(columns=[TARGET])

    # Use first test record
    sample = X_test.iloc[[0]]

    # Prediction
    prediction = model.predict(sample)[0]

    # Probability
    probability = model.predict_proba(sample)[0][1]

    print("Input:")
    print(sample.to_string(index=False))

    print()
    print("Prediction:", int(prediction))
    print(
        "Lung Cancer Probability:",
        round(float(probability), 4)
    )

    print()
    print("Inference completed successfully.")


if __name__ == "__main__":
    main()
