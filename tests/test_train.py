import os
import joblib


def test_train_model_exists():

    # CI already runs:
    # python src/train.py
    #
    # Therefore this test must NOT run train.py again.
    # Otherwise MLflow creates a second run.

    model_path = "model/model.pkl"

    assert os.path.exists(
        model_path
    ), f"Model not found: {model_path}"

    # Verify the model can actually be loaded
    model = joblib.load(model_path)

    assert model is not None

    print(f"Model verified successfully: {model_path}")
