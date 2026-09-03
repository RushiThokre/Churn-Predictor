from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import jwt
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.business_metrics import calculate_customer_value
from app.database import init_db, persist_model_version, persist_prediction_request
from app.retention import recommend_retention_action

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"
LOG_PATH = BASE_DIR / "api" / "request_log.jsonl"
API_KEY = os.getenv("API_KEY", "demo-key")
JWT_SECRET = os.getenv("JWT_SECRET", "development-secret-change-me")
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
_rate_buckets: defaultdict[str, deque[float]] = defaultdict(deque)

app = FastAPI(title="Churn Predictor API", version="1.0.0")


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, (dict, list)) else {"message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content={"error": {"code": f"HTTP_{exc.status_code}", "details": detail}})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": {"code": "VALIDATION_ERROR", "details": exc.errors()}})
logger = logging.getLogger("churn_api")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(BASE_DIR / "api" / "app.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)


class CustomerFeatures(BaseModel):
    customer_id: str | None = Field(default=None, description="Optional customer identifier")
    tenure: int = Field(..., ge=0, description="Number of months the customer has been with the company")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charge amount")
    TotalCharges: float = Field(..., ge=0, description="Total charges billed so far")
    Contract: str = Field(..., description="Contract type")
    PaymentMethod: str = Field(..., description="Payment method")
    InternetService: str = Field(..., description="Internet service type")
    TechSupport: str = Field(..., description="Tech support status")
    OnlineSecurity: str = Field(..., description="Online security status")


_model: Any | None = None
_db_ready = False


def _prediction_details(probability: float, features: dict[str, Any]) -> dict[str, Any]:
    threshold = float(getattr(_model, "decision_threshold", 0.5))
    prediction = int(probability >= threshold)
    customer_value = calculate_customer_value(features, probability)
    recommendation = recommend_retention_action(probability, features)
    return {
        "customer_id": features.get("customer_id"),
        "prediction": prediction,
        "churn_label": "churn" if prediction else "stay",
        "probability": round(probability, 3),
        "churn_probability": round(probability, 3),
        "risk_level": "HIGH" if probability >= 0.7 else "MEDIUM" if probability >= 0.4 else "LOW",
        "expected_remaining_months": customer_value["expected_remaining_months"],
        "potential_revenue": round(customer_value["potential_revenue"], 2),
        "revenue_at_risk": round(customer_value["revenue_at_risk"], 2),
        "recommendation_reason": recommendation["reason"],
        "recommended_action": recommendation["recommended_action"],
    }


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    if x_api_key == API_KEY:
        return
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        try:
            jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            return
        except jwt.PyJWTError:
            pass
    raise HTTPException(status_code=401, detail="Invalid or missing API key or JWT")


def enforce_rate_limit(request: Request) -> None:
    client_key = request.client.host if request.client else "unknown"
    now = time.monotonic()
    bucket = _rate_buckets[client_key]
    while bucket and now - bucket[0] >= RATE_LIMIT_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    bucket.append(now)


def require_prediction_access(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    require_api_key(x_api_key, authorization)
    enforce_rate_limit(request)


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


def _log_request(request: Request, payload: dict[str, Any] | list[dict[str, Any]], response: Any) -> None:
    if _db_ready:
        persist_prediction_request(payload, response)
        logger.info("Prediction request persisted: %s", request.url.path)
        return
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
    global _model, _db_ready
    try:
        init_db()
        _db_ready = True
    except Exception as error:
        logger.warning("Database unavailable; using legacy JSONL logging: %s", error)
    try:
        _model = load_model()
        named_steps = getattr(_model, "named_steps", {})
        classifier = named_steps.get("classifier") if hasattr(named_steps, "get") else None
        if _db_ready:
            persist_model_version(
                classifier.__class__.__name__ if classifier is not None else _model.__class__.__name__,
                str(MODEL_PATH),
                float(getattr(_model, "decision_threshold", 0.5)),
            )
    except Exception:
        _model = None


@app.get("/health")
@app.get("/api/v1/health")
def health() -> dict[str, str]:
    status = "ready" if _model is not None else "missing_model"
    return {"status": status}


@app.get("/metrics")
@app.get("/api/v1/metrics")
def metrics() -> dict[str, Any]:
    """Return lightweight operational metrics for the prediction service."""
    prediction_count = 0
    probabilities: list[float] = []
    if LOG_PATH.exists():
        with LOG_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    response = json.loads(line).get("response", [])
                    responses = response if isinstance(response, list) else [response]
                    for item in responses:
                        if isinstance(item, dict) and "probability" in item:
                            prediction_count += 1
                            probabilities.append(float(item["probability"]))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
    return {
        "prediction_count": prediction_count,
        "average_probability": round(sum(probabilities) / len(probabilities), 4) if probabilities else 0.0,
        "high_risk_count": sum(probability >= 0.7 for probability in probabilities),
        "model_ready": _model is not None,
    }


@app.get("/model-info")
@app.get("/api/v1/model-info")
def model_info() -> dict[str, Any]:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is not available")
    named_steps = getattr(_model, "named_steps", {})
    classifier = named_steps.get("classifier") if hasattr(named_steps, "get") else None
    return {
        "model": classifier.__class__.__name__ if classifier is not None else _model.__class__.__name__,
        "decision_threshold": float(getattr(_model, "decision_threshold", 0.5)),
        "features": list(named_steps.get("preprocessor").feature_names_in_) if named_steps.get("preprocessor") is not None and hasattr(named_steps.get("preprocessor"), "feature_names_in_") else [],
        "artifact_updated_at": datetime.fromtimestamp(MODEL_PATH.stat().st_mtime, tz=timezone.utc).isoformat() if MODEL_PATH.exists() else None,
    }


@app.post("/predict")
@app.post("/api/v1/predict")
def predict(request: Request, features: CustomerFeatures, _: None = Depends(require_prediction_access)) -> dict[str, Any]:
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
    response = _prediction_details(probability, features.model_dump())
    if response["customer_id"] is None:
        response["customer_id"] = "CUST-1"
    _log_request(request, features.model_dump(), response)
    return response


@app.post("/predict/batch")
@app.post("/api/v1/batch-predict")
@app.post("/api/v1/predict/batch")
def predict_batch(request: Request, features: list[CustomerFeatures], _: None = Depends(require_prediction_access)) -> list[dict[str, Any]]:
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
    response = [
        {**_prediction_details(float(probability), item.model_dump()), "customer_id": item.customer_id or f"CUST-{index + 1}"}
        for index, (item, probability) in enumerate(zip(features, probabilities))
    ]
    _log_request(request, [item.model_dump() for item in features], response)
    return response
