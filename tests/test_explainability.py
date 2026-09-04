import pandas as pd

from app.explainability import explain_prediction, shap_waterfall, summarize_churn_drivers
from model.train_pipeline import build_pipeline


def test_explain_prediction_returns_top_risk_drivers():
    model = build_pipeline()
    sample = pd.DataFrame(
        [{
            "tenure": 5,
            "MonthlyCharges": 80,
            "TotalCharges": 350,
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "InternetService": "Fiber optic",
            "TechSupport": "No",
            "OnlineSecurity": "No",
        }]
    )

    explanation = explain_prediction(model, sample)

    assert not explanation.empty
    assert {"feature", "contribution", "share_pct"}.issubset(explanation.columns)
    assert explanation["share_pct"].sum() > 0
    assert explanation["feature"].nunique() >= 1
    assert explanation["contribution"].gt(0).any()


def test_shap_waterfall_returns_figure():
    model = build_pipeline()
    sample = pd.DataFrame([{
        "tenure": 5,
        "MonthlyCharges": 80,
        "TotalCharges": 350,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "InternetService": "Fiber optic",
        "TechSupport": "No",
        "OnlineSecurity": "No",
    }])

    figure = shap_waterfall(model, sample)

    assert figure is not None


def test_summarize_churn_drivers_returns_ranked_impacts():
    model = build_pipeline()
    sample = pd.DataFrame([{
        "tenure": 5,
        "MonthlyCharges": 80,
        "TotalCharges": 350,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "InternetService": "Fiber optic",
        "TechSupport": "No",
        "OnlineSecurity": "No",
    }] * 3)

    drivers = summarize_churn_drivers(model, sample)

    assert not drivers.empty
    assert {"label", "impact", "share_pct"}.issubset(drivers.columns)
    assert drivers["impact"].is_monotonic_decreasing
