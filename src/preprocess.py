import os
import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = "data/survey_lung_cancer.csv"
TRAIN_FILE = "artifacts/train.csv"
TEST_FILE = "artifacts/test.csv"

TARGET = "LUNG_CANCER"


def preprocess_data():

    # Create artifacts directory
    os.makedirs("artifacts", exist_ok=True)

    # Read dataset
    print(f"Reading dataset: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print("Original shape:", df.shape)
    print("Original columns:", df.columns.tolist())

    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove unnecessary ID column
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Convert categorical columns
    for column in df.columns:

        if df[column].dtype == "object":

            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
                .str.upper()
            )

            df[column] = df[column].replace({
                "YES": 1,
                "NO": 0,
                "M": 1,
                "F": 0
            })

    # Make sure target exists
    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    # Convert target
    if df[TARGET].dtype == "object":

        df[TARGET] = (
            df[TARGET]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({
                "YES": 1,
                "NO": 0
            })
        )

    # Convert all remaining columns to numeric
    for column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove missing values
    before = len(df)

    df = df.dropna()

    after = len(df)

    print(f"Removed {before - after} rows with missing values.")

    # Separate features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print("Features:", X.shape)
    print("Target:", y.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create train dataset
    train = X_train.copy()
    train[TARGET] = y_train

    # Create test dataset
    test = X_test.copy()
    test[TARGET] = y_test

    # Save datasets
    train.to_csv(
        TRAIN_FILE,
        index=False
    )

    test.to_csv(
        TEST_FILE,
        index=False
    )

    print("\nPreprocessing completed.")

    print("Train shape:", train.shape)
    print("Test shape :", test.shape)

    print("\nSaved:")
    print(TRAIN_FILE)
    print(TEST_FILE)


if __name__ == "__main__":
    preprocess_data()
