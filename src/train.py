import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRAIN_FILE = "artifacts/train.csv"
MODEL_FILE = "model/model.pkl"

TARGET = "LUNG_CANCER"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://32.198.38.148:5000/"
)

EXPERIMENT_NAME = "LungCancerPrediction"


# --------------------------------------------------
# Create directories
# --------------------------------------------------

os.makedirs("model", exist_ok=True)


# --------------------------------------------------
# Configure MLflow
# --------------------------------------------------

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    EXPERIMENT_NAME
)


print("MLflow Tracking URI:")
print(MLFLOW_TRACKING_URI)

print("MLflow Experiment:")
print(EXPERIMENT_NAME)


# --------------------------------------------------
# Load training data
# --------------------------------------------------

print("\nLoading training data...")

train = pd.read_csv(TRAIN_FILE)

print("Training data shape:", train.shape)


# --------------------------------------------------
# Separate features and target
# --------------------------------------------------

if TARGET not in train.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )

X_train = train.drop(
    columns=[TARGET]
)

y_train = train[TARGET]


print("Feature shape:", X_train.shape)
print("Target shape:", y_train.shape)


# --------------------------------------------------
# Model parameters
# --------------------------------------------------

max_iter = 1000
random_state = 42


# --------------------------------------------------
# Start MLflow run
# --------------------------------------------------

with mlflow.start_run() as run:

    print("\nMLflow Run ID:")
    print(run.info.run_id)

    # ----------------------------------------------
    # Create model
    # ----------------------------------------------

    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state
    )

    # ----------------------------------------------
    # Train model
    # ----------------------------------------------

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )

    # ----------------------------------------------
    # Predictions
    # ----------------------------------------------

    predictions = model.predict(
        X_train
    )

    # ----------------------------------------------
    # Training accuracy
    # ----------------------------------------------

    accuracy = accuracy_score(
        y_train,
        predictions
    )

    print(
        "Training accuracy:",
        accuracy
    )

    # ----------------------------------------------
    # Log parameters
    # ----------------------------------------------

    mlflow.log_param(
        "model_type",
        "LogisticRegression"
    )

    mlflow.log_param(
        "max_iter",
        max_iter
    )

    mlflow.log_param(
        "random_state",
        random_state
    )

    mlflow.log_param(
        "training_rows",
        X_train.shape[0]
    )

    mlflow.log_param(
        "feature_count",
        X_train.shape[1]
    )

    # ----------------------------------------------
    # Log metrics
    # ----------------------------------------------

    mlflow.log_metric(
        "training_accuracy",
        accuracy
    )

    # ----------------------------------------------
    # Save model locally
    # ----------------------------------------------

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        "\nModel saved to:",
        MODEL_FILE
    )

    # ----------------------------------------------
    # Log model to MLflow
    # ----------------------------------------------

    mlflow.sklearn.log_model(
        model,
        name="lung_cancer_model"
    )

    print(
        "\nModel logged to MLflow."
    )

    print(
        "Run completed successfully."
    )
