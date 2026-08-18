import pandas as pd
from sklearn.model_selection import train_test_split

# Read dataset
df = pd.read_csv("data/survey_lung_cancer.csv")

print("Original shape:", df.shape)

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Convert categorical values
df["GENDER"] = df["GENDER"].map({
    "M": 1,
    "F": 0
})

df["LUNG_CANCER"] = df["LUNG_CANCER"].map({
    "YES": 1,
    "NO": 0
})

# Convert all columns to numbers
df = df.apply(pd.to_numeric)

# Split features and target
X = df.drop("LUNG_CANCER", axis=1)
y = df["LUNG_CANCER"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Add target column back
train = X_train.copy()
train["LUNG_CANCER"] = y_train

test = X_test.copy()
test["LUNG_CANCER"] = y_test

# Save files
train.to_csv("artifacts/train.csv", index=False)
test.to_csv("artifacts/test.csv", index=False)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Preprocessing completed!")