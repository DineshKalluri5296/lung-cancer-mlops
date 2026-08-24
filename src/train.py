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

from mlflow.models import infer_signature


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRAIN_DATA_PATH = "artifacts/train.csv"
TEST_DATA_PATH = "artifacts/test.csv"

MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://44.201.14.6:5000"
)

EXPERIMENT_NAME = "LungCancerPrediction23"

REGISTERED_MODEL_NAME = "LungCancerModel"

ALGORITHM = os.getenv(
    "ALGORITHM",
    "logistic_regression"
)


# --------------------------------------------------
# Create model
# --------------------------------------------------

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


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 60)
    print("LUNG CANCER MODEL TRAINING")
    print("=" * 60)

    print(f"Algorithm: {ALGORITHM}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Registered Model: {REGISTERED_MODEL_NAME}")

    # --------------------------------------------------
    # MLflow configuration
    # --------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print(
        f"MLflow Tracking URI: "
        f"{MLFLOW_TRACKING_URI}"
    )

    # --------------------------------------------------
    # Load training data
    # --------------------------------------------------

    print("\nLoading training data...")

    train_df = pd.read_csv(
        TRAIN_DATA_PATH
    )

    print(
        f"Training data shape: "
        f"{train_df.shape}"
    )

    # --------------------------------------------------
    # Load test data
    # --------------------------------------------------

    print("\nLoading test data...")

    test_df = pd.read_csv(
        TEST_DATA_PATH
    )

    print(
        f"Test data shape: "
        f"{test_df.shape}"
    )

    # --------------------------------------------------
    # Separate features and target
    # --------------------------------------------------

    target_column = "LUNG_CANCER"

    X_train = train_df.drop(
        target_column,
        axis=1
    )

    y_train = train_df[
        target_column
    ]

    X_test = test_df.drop(
        target_column,
        axis=1
    )

    y_test = test_df[
        target_column
    ]

    print(
        f"\nTraining features: "
        f"{X_train.shape}"
    )

    print(
        f"Training target: "
        f"{y_train.shape}"
    )

    print(
        f"Test features: "
        f"{X_test.shape}"
    )

    print(
        f"Test target: "
        f"{y_test.shape}"
    )

    # --------------------------------------------------
    # Create model
    # --------------------------------------------------

    model = create_model()

    print(
        f"\nCreated model: "
        f"{model.__class__.__name__}"
    )

    # --------------------------------------------------
    # Start MLflow run
    # --------------------------------------------------

    with mlflow.start_run() as run:

        run_id = run.info.run_id

        print(
            f"\nMLflow Run ID: "
            f"{run_id}"
        )

        # --------------------------------------------------
        # MLflow tags
        # --------------------------------------------------

        mlflow.set_tag(
            "project",
            "lung-cancer-mlops"
        )

        mlflow.set_tag(
            "algorithm",
            ALGORITHM
        )

        mlflow.set_tag(
            "dataset",
            "survey_lung_cancer.csv"
        )

        mlflow.set_tag(
            "environment",
            "ci"
        )

        # --------------------------------------------------
        # Log basic parameters
        # --------------------------------------------------

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
            len(X_train)
        )

        mlflow.log_param(
            "test_rows",
            len(X_test)
        )

        mlflow.log_param(
            "features",
            X_train.shape[1]
        )

        # --------------------------------------------------
        # Log model parameters
        # --------------------------------------------------

        mlflow.log_params(
            model.get_params()
        )

        # --------------------------------------------------
        # Train
        # --------------------------------------------------

        print("\nTraining model...")

        model.fit(
            X_train,
            y_train
        )

        print("Training completed.")

        # --------------------------------------------------
        # Prediction on TEST data
        # --------------------------------------------------

        print("\nEvaluating model on test data...")

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        # --------------------------------------------------
        # Calculate metrics
        # --------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities
        )

        # --------------------------------------------------
        # Log metrics
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Create model directory
        # --------------------------------------------------

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        # --------------------------------------------------
        # Save local model
        # --------------------------------------------------

        joblib.dump(
            model,
            MODEL_PATH
        )

        print(
            f"\nLocal model saved to: "
            f"{MODEL_PATH}"
        )

        # --------------------------------------------------
        # MLflow model signature
        # --------------------------------------------------

        predictions_for_signature = model.predict(
            X_test.head(5)
        )

        signature = infer_signature(
            X_test,
            predictions_for_signature
        )

        # --------------------------------------------------
        # MLflow input example
        # --------------------------------------------------

        input_example = X_test.head(1)

        # --------------------------------------------------
        # Log + Register model
        # --------------------------------------------------

        print(
            "\nLogging model to MLflow..."
        )

        print(
            f"Registering as: "
            f"{REGISTERED_MODEL_NAME}"
        )

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME
        )

        # --------------------------------------------------
        # Print results
        # --------------------------------------------------

        print("\n" + "=" * 60)
        print("MODEL RESULTS")
        print("=" * 60)

        print(
            f"Algorithm : {ALGORITHM}"
        )

        print(
            f"Accuracy  : {accuracy:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1 Score  : {f1:.4f}"
        )

        print(
            f"ROC-AUC   : {roc_auc:.4f}"
        )

        print(
            f"Run ID    : {run_id}"
        )

        print(
            f"Model     : {REGISTERED_MODEL_NAME}"
        )

        print(
            f"Model URI : {model_info.model_uri}"
        )

        print("=" * 60)

        print(
            "\nTraining and registration completed successfully."
        )


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()
