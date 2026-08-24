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

DATA_PATH = "artifacts/train.csv"
MODEL_PATH = "model/model.pkl"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://32.198.38.148:5000"
)

EXPERIMENT_NAME = "LungCancerPrediction23"
REGISTERED_MODEL_NAME = "LungCancerModel"

ALGORITHM = os.getenv("ALGORITHM", "logistic_regression")


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
        raise ValueError(f"Unsupported algorithm: {ALGORITHM}")


def main():

    print("=" * 60)
    print(f"Algorithm: {ALGORITHM}")
    print("=" * 60)

    # Create local model directory
    os.makedirs("model", exist_ok=True)

    # MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load data
    df = pd.read_csv(DATA_PATH)

    X = df.drop("LUNG_CANCER", axis=1)
    y = df["LUNG_CANCER"]

    model = create_model()

    # ONE execution = ONE MLflow run
    with mlflow.start_run() as run:

        run_id = run.info.run_id

        print(f"MLflow Run ID: {run_id}")

        # Basic metadata
        mlflow.log_param("algorithm", ALGORITHM)
        mlflow.log_param("dataset", "survey_lung_cancer.csv")
        mlflow.log_param("training_rows", len(df))
        mlflow.log_param("features", X.shape[1])

        # Train
        model.fit(X, y)

        # Predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        # Metrics
        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, zero_division=0)
        recall = recall_score(y, predictions, zero_division=0)
        f1 = f1_score(y, predictions, zero_division=0)
        roc_auc = roc_auc_score(y, probabilities)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log model parameters
        mlflow.log_params(model.get_params())

        # Save local model
        joblib.dump(model, MODEL_PATH)

        print(f"Model saved locally: {MODEL_PATH}")

        # Register ONLY ONCE
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME
        )

        print("=" * 60)
        print("Training completed")
        print(f"Run ID: {run_id}")
        print(f"Algorithm: {ALGORITHM}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Registered Model: {REGISTERED_MODEL_NAME}")
        print("=" * 60)


if __name__ == "__main__":
    main()
