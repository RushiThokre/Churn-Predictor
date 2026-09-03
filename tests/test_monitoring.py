import json

import pandas as pd

from app.monitoring import build_feature_drift_report, build_monitoring_summary, calculate_drift, load_feedback, load_prediction_events, simulate_drift, write_feedback
from model.train_pipeline import build_pipeline


def test_monitoring_summary_builds_trends(tmp_path):
    log_path = tmp_path / "request_log.jsonl"
    records = [
        {"timestamp": "2026-08-30T10:00:00+00:00", "path": "/predict", "response": {"prediction": 1, "probability": 0.8, "churn_label": "churn"}},
        {"timestamp": "2026-08-30T11:00:00+00:00", "path": "/predict/batch", "response": [{"prediction": 0, "probability": 0.2, "churn_label": "stay"}]},
    ]
    log_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    events = load_prediction_events(log_path)
    summary = build_monitoring_summary(events)

    assert summary["volume"] == 2
    assert len(summary["trend"]) == 1
    assert summary["trend"].iloc[0]["prediction_volume"] == 2
    assert summary["confidence"]["predictions"].sum() == 2


def test_feedback_and_drift_outputs(tmp_path):
    feedback_path = tmp_path / "feedback.jsonl"
    write_feedback(1, 0.9, True, feedback_path)
    feedback = load_feedback(feedback_path)

    assert feedback.iloc[0]["feedback"] == "correct"
    assert feedback.iloc[0]["prediction"] == 1

    model = build_pipeline()
    source = pd.read_csv("data/telco.csv").rename(columns={
        "Tenure in Months": "tenure",
        "Monthly Charge": "MonthlyCharges",
        "Total Charges": "TotalCharges",
        "Payment Method": "PaymentMethod",
        "Internet Service": "InternetService",
        "Premium Tech Support": "TechSupport",
        "Online Security": "OnlineSecurity",
    })
    source["tenure_bucket"] = pd.cut(source["tenure"], bins=[0, 12, 24, 48, 72], labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"])
    drift = simulate_drift(model, source, sample_size=5)
    assert len(drift) == 5
    assert {"probability", "cohort"}.issubset(drift.columns)


def test_feature_drift_report_includes_psi_ks_and_js():
    reference = pd.DataFrame({"MonthlyCharges": [40, 45, 50, 55], "Contract": ["One Year"] * 4})
    current = pd.DataFrame({"MonthlyCharges": [140, 145, 150, 155], "Contract": ["Month-to-Month"] * 4})

    report = build_feature_drift_report(reference, current, ["MonthlyCharges", "Contract"])

    assert set(report["feature"]) == {"MonthlyCharges", "Contract"}
    assert {"psi", "ks_statistic", "ks_pvalue", "js_divergence", "status"}.issubset(report.columns)
    assert (report["status"] == "drift").any()


def test_numeric_drift_is_stable_for_matching_distributions():
    scores = calculate_drift(pd.Series(range(100)), pd.Series(range(100)))
    assert scores["status"] == "stable"
    assert scores["js_divergence"] == 0
