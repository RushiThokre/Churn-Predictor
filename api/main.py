from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"
LOG_PATH = BASE_DIR / "api" / "request_log.jsonl"
API_KEY = os.getenv("API_KEY", "demo-key")

app = FastAPI(title="Churn Predictor API", version="1.0.0")
logger = logging.getLogger("churn_api")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(BASE_DIR / "api" / "app.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)


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


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if x_api_key is None or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _normalize_batch_frame(payload: pd.DataFrame) -> pd.DataFrame:
    payload = payload.copy()
    payload["tenure_bucket"] = pd.cut(
        payload["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
    )
    payload["Contract"] = payload["Contract"].map({
        "Month-to-month": "Month-to-Month",
        "Month-to-Month": "Month-to-Month",
        "One year": "One Year",
        "One Year": "One Year",
        "Two year": "Two Year",
        "Two Year": "Two Year",
    }).fillna(payload["Contract"])
    payload["PaymentMethod"] = payload["PaymentMethod"].map({
        "Electronic check": "Electronic Check",
        "Electronic Check": "Electronic Check",
        "Mailed check": "Mailed Check",
        "Mailed Check": "Mailed Check",
        "Bank transfer": "Bank Transfer",
        "Bank Transfer": "Bank Transfer",
        "Credit card": "Credit Card",
        "Credit Card": "Credit Card",
    }).fillna(payload["PaymentMethod"])
    payload["InternetService"] = payload["InternetService"].map({
        "Fiber optic": "Fiber Optic",
        "Fiber Optic": "Fiber Optic",
        "DSL": "DSL",
        "No": "No",
    }).fillna(payload["InternetService"])
    return payload


def _log_request(request: Request, payload: dict[str, Any], response: Any) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "path": request.url.path,
        "method": request.method,
        "request": payload,
        "response": response,
    }
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    logger.info("Prediction request logged: %s", request.url.path)


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
def predict(request: Request, features: CustomerFeatures, _: None = Depends(require_api_key)) -> dict[str, Any]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Train and save model/churn_model.pkl first.")

    payload = pd.DataFrame([features.model_dump()])
    payload = _normalize_batch_frame(payload)
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
    response = {
        "prediction": prediction,
        "churn_label": "churn" if prediction == 1 else "stay",
        "probability": round(probability, 3),
    }
    _log_request(request, features.model_dump(), response)
    return response


@app.post("/predict/batch")
def predict_batch(request: Request, features: list[CustomerFeatures], _: None = Depends(require_api_key)) -> list[dict[str, Any]]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not available. Train and save model/churn_model.pkl first.")
    if not features:
        raise HTTPException(status_code=400, detail="At least one customer record is required.")

    payload = pd.DataFrame([item.model_dump() for item in features])
    payload = _normalize_batch_frame(payload)
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
    probabilities = _model.predict_proba(model_input)[:, 1]
    predictions = _model.predict(model_input)
    response = [
        {
            "prediction": int(prediction),
            "churn_label": "churn" if int(prediction) == 1 else "stay",
            "probability": round(float(probability), 3),
        }
        for prediction, probability in zip(predictions, probabilities)
    ]
    _log_request(request, [item.model_dump() for item in features], response)
    return response
