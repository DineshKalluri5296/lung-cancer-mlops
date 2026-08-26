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
    "http://100.60.81.211:5000"
)

EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT",
    "LungCancerPrediction212"
)

REGISTERED_MODEL_NAME = "LungCancerModelprediction"

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
    # Disable autologging so ONLY our explicit run is created.
    mlflow.autolog(disable=True)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print(f"Reading dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    X = df.drop(
        "LUNG_CANCER",
        axis=1
    )

    y = df["LUNG_CANCER"]

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model()

    # ========================================================
    # ONE AND ONLY ONE MLFLOW RUN
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
        # Run tags
        # ----------------------------------------------------

        mlflow.set_tag(
            "algorithm",
            ALGORITHM
        )

        mlflow.set_tag(
            "project",
            "lung-cancer-mlops"
        )

        mlflow.set_tag(
            "dataset",
            "survey_lung_cancer.csv"
        )

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
        # --------------------------------------------------------

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
        # DO NOT register here.
        # Registration happens after the run finishes.
        # ----------------------------------------------------

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model"
        )

        print(
            f"Model logged: {model_info.model_uri}"
        )

    # ========================================================
    # MLFLOW RUN FINISHED
    # ========================================================

    print()
    print("=" * 70)
    print("MLFLOW RUN FINISHED")
    print("=" * 70)

    print(f"Run ID: {run_id}")

    # ========================================================
    # REGISTER MODEL
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

    except Exception:

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

    version = model_version.version

    # ========================================================
    # ADD MODEL VERSION DESCRIPTION
    # ========================================================

    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=version,
        description=(
            f"Lung Cancer model trained using "
            f"{ALGORITHM}. "
            f"Accuracy={accuracy:.4f}, "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"F1={f1:.4f}, "
            f"ROC-AUC={roc_auc:.4f}. "
            f"MLflow Run ID={run_id}."
        )
    )

    # ========================================================
    # ADD MODEL VERSION TAGS
    # ========================================================

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="algorithm",
        value=ALGORITHM
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="project",
        value="lung-cancer-mlops"
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="accuracy",
        value=f"{accuracy:.4f}"
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="precision",
        value=f"{precision:.4f}"
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="recall",
        value=f"{recall:.4f}"
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="f1",
        value=f"{f1:.4f}"
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="roc_auc",
        value=f"{roc_auc:.4f}"
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=version,
        key="run_id",
        value=run_id
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

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
        f"{version}"
    )

    print(
        f"Run ID           : "
        f"{run_id}"
    )

    print(
        f"Algorithm        : "
        f"{ALGORITHM}"
    )

    print(
        f"Accuracy         : "
        f"{accuracy:.4f}"
    )

    print(
        f"F1               : "
        f"{f1:.4f}"
    )

    print(
        f"ROC-AUC          : "
        f"{roc_auc:.4f}"
    )

    print(
        f"S3 Model Path    : "
        f"s3://$S3_BUCKET/models/{ALGORITHM}/model.pkl"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
