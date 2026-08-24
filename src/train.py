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
        raise ValueError(
            f"Unsupported algorithm: {ALGORITHM}"
        )


def main():

    print(f"Algorithm: {ALGORITHM}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(DATA_PATH)

    X = df.drop("LUNG_CANCER", axis=1)
    y = df["LUNG_CANCER"]

    model = create_model()

    with mlflow.start_run() as run:

        mlflow.log_param(
            "algorithm",
            ALGORITHM
        )

        model.fit(X, y)

        predictions = model.predict(X)

        probabilities = model.predict_proba(X)[:, 1]

        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        f1 = f1_score(y, predictions)
        roc_auc = roc_auc_score(y, probabilities)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        mlflow.log_params(model.get_params())

        joblib.dump(model, MODEL_PATH)

        mlflow.sklearn.log_model(
            model,
            "model"
        )

        print("MLflow Run ID:", run.info.run_id)
        print("Algorithm:", ALGORITHM)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
        print("ROC-AUC:", roc_auc)

        print("Model saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
