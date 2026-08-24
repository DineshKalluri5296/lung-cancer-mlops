import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient

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
MODEL_PATH = "model/model.pkl"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://44.201.14.6:5000"
)

EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT",
    "LungCancerPrediction2311"
)

REGISTERED_MODEL_NAME = "LungCancerModel6911"

ALGORITHM = os.getenv(
    "ALGORITHM",
    "logistic_regression"
)


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

    print("=" * 70)
    print("LUNG CANCER MODEL TRAINING")
    print("=" * 70)

    print(f"Algorithm    : {ALGORITHM}")
    print(f"Experiment   : {EXPERIMENT_NAME}")
    print(f"Tracking URI : {MLFLOW_TRACKING_URI}")
    print(f"Model        : {REGISTERED_MODEL_NAME}")
    print("=" * 70)

    # --------------------------------------------------------
    # Create model directory
    # --------------------------------------------------------

    os.makedirs("model", exist_ok=True)

    # --------------------------------------------------------
    # Configure MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    # IMPORTANT:
    # Make sure autologging is disabled
    mlflow.autolog(disable=True)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    X = df.drop(
        "LUNG_CANCER",
        axis=1
    )

    y = df["LUNG_CANCER"]

    model = create_model()

    # ========================================================
    # ONE AND ONLY ONE MLflow RUN
    # ========================================================

    with mlflow.start_run(
        run_name=ALGORITHM
    ) as run:

        run_id = run.info.run_id

        print()
        print("=" * 70)
        print("MLFLOW RUN CREATED")
        print(f"Run ID: {run_id}")
        print("=" * 70)

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "algorithm",
            ALGORITHM
        )

        mlflow.log_param(
            "dataset",
            "survey_lung_cancer.csv"
        )

        mlflow.log_param(
            "training_rows",
            len(df)
        )

        mlflow.log_param(
            "features",
            X.shape[1]
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        print("Training model...")

        model.fit(X, y)

        # ----------------------------------------------------
        # Predictions
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
        # Log metrics
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
        # Log algorithm parameters
        # ----------------------------------------------------

        mlflow.log_params(
            model.get_params()
        )

        # ----------------------------------------------------
        # Save local model
        # ----------------------------------------------------

        joblib.dump(
            model,
            MODEL_PATH
        )

        print(
            f"Local model saved: {MODEL_PATH}"
        )

        # ----------------------------------------------------
        # Log model ONLY
        #
        # DO NOT use registered_model_name here
        # ----------------------------------------------------

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model"
        )

        print(
            f"Model logged: {model_info.model_uri}"
        )

        print()
        print("=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70)

        print(f"Run ID    : {run_id}")
        print(f"Algorithm : {ALGORITHM}")
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1        : {f1:.4f}")
        print(f"ROC-AUC   : {roc_auc:.4f}")

    # ========================================================
    # REGISTER MODEL AFTER RUN HAS FINISHED
    # ========================================================

    print()
    print("=" * 70)
    print("REGISTERING MODEL")
    print("=" * 70)

    client = MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    # --------------------------------------------------------
    # Create registered model if it doesn't exist
    # --------------------------------------------------------

    try:

        client.create_registered_model(
            REGISTERED_MODEL_NAME
        )

        print(
            f"Created registered model: "
            f"{REGISTERED_MODEL_NAME}"
        )

    except Exception as e:

        # Model already exists
        print(
            f"Registered model already exists: "
            f"{REGISTERED_MODEL_NAME}"
        )

    # --------------------------------------------------------
    # Register THIS exact run's model
    # --------------------------------------------------------

    model_uri = f"runs:/{run_id}/model"

    model_version = client.create_model_version(
        name=REGISTERED_MODEL_NAME,
        source=model_uri,
        run_id=run_id
    )

    print()
    print("=" * 70)
    print("MODEL REGISTERED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"Registered Model : "
        f"{REGISTERED_MODEL_NAME}"
    )

    print(
        f"Version          : "
        f"{model_version.version}"
    )

    print(
        f"Run ID           : "
        f"{run_id}"
    )

    print(
        f"Algorithm        : "
        f"{ALGORITHM}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
