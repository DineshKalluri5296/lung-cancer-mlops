import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "artifacts/train.csv"

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")


MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://32.198.38.148:5000"
)

# IMPORTANT:
# All algorithms use the SAME MLflow experiment
EXPERIMENT_NAME = "LungCancerPrediction23"

# GitHub Actions will provide this value
ALGORITHM = os.getenv(
    "ALGORITHM",
    "logistic_regression"
)


# ============================================================
# Create model directory
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# Create model
# ============================================================

def create_model():

    if ALGORITHM == "logistic_regression":

        return LogisticRegression(
            max_iter=1000,
            random_state=42
        )

    elif ALGORITHM == "random_forest":

        return RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

    elif ALGORITHM == "svm":

        return SVC(
            probability=True,
            kernel="rbf",
            random_state=42
        )

    else:

        raise ValueError(
            f"Unsupported algorithm: {ALGORITHM}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("LUNG CANCER MODEL TRAINING")
    print("=" * 60)

    print("Algorithm:", ALGORITHM)
    print("MLflow Experiment:", EXPERIMENT_NAME)
    print("MLflow Tracking URI:", MLFLOW_TRACKING_URI)

    # --------------------------------------------------------
    # MLflow configuration
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    print("Loading training data...")

    df = pd.read_csv(DATA_PATH)

    print("Training data shape:", df.shape)

    X = df.drop(
        "LUNG_CANCER",
        axis=1
    )

    y = df["LUNG_CANCER"]

    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model()

    print("Model:", model)

    # --------------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run() as run:

        print("MLflow Run ID:", run.info.run_id)

        # ----------------------------------------------------
        # Log algorithm
        # ----------------------------------------------------

        mlflow.log_param(
            "algorithm",
            ALGORITHM
        )

        # ----------------------------------------------------
        # Log dataset information
        # ----------------------------------------------------

        mlflow.log_param(
            "dataset",
            "survey_lung_cancer.csv"
        )

        mlflow.log_param(
            "training_rows",
            X.shape[0]
        )

        mlflow.log_param(
            "features",
            X.shape[1]
        )

        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        print("Training model...")

        model.fit(
            X,
            y
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predictions = model.predict(X)

        probabilities = model.predict_proba(X)[:, 1]

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y,
            predictions
        )

        precision = precision_score(
            y,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y,
            probabilities
        )

        # ----------------------------------------------------
        # Log metrics to MLflow
        # ----------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )

        # ----------------------------------------------------
        # Log model parameters
        # ----------------------------------------------------

        mlflow.log_params(
            model.get_params()
        )

        # ----------------------------------------------------
        # Save model locally
        # ----------------------------------------------------

        joblib.dump(
            model,
            MODEL_PATH
        )

        print(
            "Model saved:",
            MODEL_PATH
        )

        # ----------------------------------------------------
        # Log model to MLflow
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print("=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)

        print("MLflow Run ID:", run.info.run_id)
        print("Algorithm:", ALGORITHM)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
        print("ROC-AUC:", roc_auc)

        print("Local model:", MODEL_PATH)

        print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
