import os

import numpy as np
from fastapi.testclient import TestClient

import api.main as api


class DummyModel:
    def predict_proba(self, payload):
        prob = np.array([0.85, 0.2])
        return np.column_stack([1 - prob, prob])[: len(payload)]

    def predict(self, payload):
        return np.array([1, 0])[: len(payload)]


def setup_function():
    api._model = DummyModel()
    api.API_KEY = "test-key"
    os.environ["API_KEY"] = "test-key"


def test_predict_requires_api_key():
    client = TestClient(api.app)
    response = client.post("/predict", json={
        "tenure": 12,
        "MonthlyCharges": 70,
        "TotalCharges": 840,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "InternetService": "Fiber optic",
        "TechSupport": "No",
        "OnlineSecurity": "No",
    })
    assert response.status_code == 401


def test_predict_batch_returns_all_predictions():
    client = TestClient(api.app)
    response = client.post(
        "/predict/batch",
        json=[
            {
                "tenure": 12,
                "MonthlyCharges": 70,
                "TotalCharges": 840,
                "Contract": "Month-to-month",
                "PaymentMethod": "Electronic check",
                "InternetService": "Fiber optic",
                "TechSupport": "No",
                "OnlineSecurity": "No",
            },
            {
                "tenure": 36,
                "MonthlyCharges": 45,
                "TotalCharges": 1600,
                "Contract": "One year",
                "PaymentMethod": "Bank transfer",
                "InternetService": "DSL",
                "TechSupport": "Yes",
                "OnlineSecurity": "Yes",
            },
        ],
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    assert payload[0]["churn_label"] in {"churn", "stay"}
    assert payload[0]["probability"] >= 0
