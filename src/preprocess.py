import pandas as pd
from sklearn.model_selection import train_test_split
import os

INPUT_FILE = "data/survey_lung_cancer.csv"
TRAIN_FILE = "artifacts/train.csv"
TEST_FILE = "artifacts/test.csv"

TARGET = "LUNG_CANCER"


# Create artifacts folder
os.makedirs("artifacts", exist_ok=True)


# Read dataset
df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())


# Remove spaces from column names
df.columns = df.columns.str.strip()


# Remove unnecessary column if present
if "id" in df.columns:
    df = df.drop(columns=["id"])


# Convert text values to numbers
for column in df.columns:

    if df[column].dtype == "object":

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        df[column] = df[column].map({
            "YES": 1,
            "NO": 0,
            "M": 1,
            "F": 0
        })


# Remove rows with missing values
df = df.dropna()


# Make sure target exists
if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )


# Separate features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]


# Convert target if necessary
if y.dtype == "object":
    y = (
        y.astype(str)
        .str.strip()
        .str.upper()
        .map({
            "YES": 1,
            "NO": 0
        })
    )


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create final datasets
train = X_train.copy()
train[TARGET] = y_train

test = X_test.copy()
test[TARGET] = y_test


# Save
train.to_csv(TRAIN_FILE, index=False)
test.to_csv(TEST_FILE, index=False)


print("\nPreprocessing completed.")

print("Train shape:", train.shape)
print("Test shape :", test.shape)

print("\nSaved:")
print(TRAIN_FILE)
print(TEST_FILE)