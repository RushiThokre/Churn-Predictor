from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"

app = FastAPI(title="Churn Predictor API", version="1.0.0")


class CustomerFeatures(BaseModel):
    tenure: int = Field(..., ge=0, description="Number of months the customer has been with the company")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charge amount")
    TotalCharges: float = Field(..., ge=0, description="Total charges billed so far")
    Contract: str = Field(..., description="Contract type")
    PaymentMethod: str = Field(..., description="Payment method")
    InternetService: str = Field(..., description="Internet service type")
    TechSupport: str = Field(..., description="Tech support status")
    OnlineSecurity: str = Field(..., description="Online security status")


_model: Any | None = None


def load_model() -> Any:
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@app.on_event("startup")
def startup_event() -> None:
    global _model
    try:
        _model = load_model()
    except Exception:
        _model = None


@app.get("/health")
def health() -> dict[str, str]:
    status = "ready" if _model is not None else "missing_model"
    return {"status": status}


@app.post("/predict")
def predict(features: CustomerFeatures) -> dict[str, Any]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Train and save model/churn_model.pkl first.")

    payload = pd.DataFrame([features.model_dump()])
    payload["tenure_bucket"] = pd.cut(
        payload["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
    )

    feature_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "PaymentMethod",
        "InternetService",
        "TechSupport",
        "OnlineSecurity",
        "tenure_bucket",
    ]
    model_input = payload[feature_columns]

    probability = float(_model.predict_proba(model_input)[0][1])
    prediction = int(_model.predict(model_input)[0])

    return {
        "prediction": prediction,
        "churn_label": "churn" if prediction == 1 else "stay",
        "probability": round(probability, 3),
    }
