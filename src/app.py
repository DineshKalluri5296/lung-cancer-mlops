from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "model/model.pkl"


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Lung Cancer Prediction API",
    description="Lung Cancer ML Model deployed on EKS",
    version="1.0.0"
)


# ============================================================
# Load model
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# ============================================================
# Request schema
# ============================================================

class PredictionRequest(BaseModel):
    features: dict


# ============================================================
# Root endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "application": "Lung Cancer Prediction API",
        "status": "running",
        "model": "loaded"
    }


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ============================================================
# Model information
# ============================================================

@app.get("/model")
def model_info():

    features = None

    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)

    return {
        "model_type": type(model).__name__,
        "features": features
    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        # Convert request features to DataFrame
        data = pd.DataFrame(
            [request.features]
        )

        # ----------------------------------------------------
        # Check feature names
        # ----------------------------------------------------

        if hasattr(model, "feature_names_in_"):

            expected_features = list(
                model.feature_names_in_
            )

            received_features = list(
                data.columns
            )

            missing_features = [
                feature
                for feature in expected_features
                if feature not in received_features
            ]

            if missing_features:

                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Missing features",
                        "missing_features": missing_features,
                        "expected_features": expected_features
                    }
                )

            # Ensure same order as training
            data = data[
                expected_features
            ]

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(data)[0]

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probability = None

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(
                data
            )[0][1]

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {
            "prediction": int(prediction),
            "prediction_label": (
                "Lung Cancer"
                if int(prediction) == 1
                else "No Lung Cancer"
            ),
            "probability": (
                float(probability)
                if probability is not None
                else None
            )
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
