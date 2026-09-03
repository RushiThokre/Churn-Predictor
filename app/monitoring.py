from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from app.database import read_feedback, read_prediction_events, read_prediction_inputs, persist_feedback


MONITORING_LOG_PATH = Path(__file__).resolve().parent.parent / "api" / "request_log.jsonl"
FEEDBACK_LOG_PATH = Path(__file__).resolve().parent.parent / "api" / "feedback_log.jsonl"


def calculate_drift(reference: pd.Series, current: pd.Series, bins: int = 10) -> dict[str, float | str]:
    """Calculate PSI, KS, and Jensen-Shannon drift scores for a numeric feature."""
    reference = pd.to_numeric(reference, errors="coerce").dropna()
    current = pd.to_numeric(current, errors="coerce").dropna()
    if reference.empty or current.empty:
        return {"psi": 0.0, "ks_statistic": 0.0, "ks_pvalue": 1.0, "js_divergence": 0.0, "status": "insufficient_data"}
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return {"psi": 0.0, "ks_statistic": 0.0, "ks_pvalue": 1.0, "js_divergence": 0.0, "status": "stable"}
    expected = np.histogram(reference, bins=edges)[0] / len(reference)
    actual = np.histogram(current, bins=edges)[0] / len(current)
    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    psi = float(np.sum((actual - expected) * np.log(actual / expected)))
    midpoint = (expected + actual) / 2
    js_divergence = float(0.5 * np.sum(expected * np.log(expected / midpoint)) + 0.5 * np.sum(actual * np.log(actual / midpoint)))
    ks_statistic, ks_pvalue = ks_2samp(reference, current)
    status = "drift" if psi >= 0.25 or js_divergence >= 0.2 or ks_pvalue < 0.05 else "warning" if psi >= 0.1 or js_divergence >= 0.1 else "stable"
    return {"psi": psi, "ks_statistic": float(ks_statistic), "ks_pvalue": float(ks_pvalue), "js_divergence": js_divergence, "status": status}


def _categorical_drift(reference: pd.Series, current: pd.Series) -> dict[str, float | str]:
    reference = reference.dropna().astype(str)
    current = current.dropna().astype(str)
    if reference.empty or current.empty:
        return {"psi": 0.0, "ks_statistic": 0.0, "ks_pvalue": 1.0, "js_divergence": 0.0, "status": "insufficient_data"}
    categories = sorted(set(reference) | set(current))
    expected = reference.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(dtype=float)
    actual = current.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(dtype=float)
    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)
    expected = expected / expected.sum()
    actual = actual / actual.sum()
    psi = float(np.sum((actual - expected) * np.log(actual / expected)))
    midpoint = (expected + actual) / 2
    js_divergence = float(0.5 * np.sum(expected * np.log(expected / midpoint)) + 0.5 * np.sum(actual * np.log(actual / midpoint)))
    status = "drift" if psi >= 0.25 or js_divergence >= 0.2 else "warning" if psi >= 0.1 or js_divergence >= 0.1 else "stable"
    return {"psi": psi, "ks_statistic": 0.0, "ks_pvalue": 1.0, "js_divergence": js_divergence, "status": status}


def build_feature_drift_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Compare reference and observed production feature distributions."""
    rows = []
    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            continue
        if pd.api.types.is_numeric_dtype(reference[feature]):
            scores = calculate_drift(reference[feature], current[feature])
        else:
            scores = _categorical_drift(reference[feature], current[feature])
        rows.append({"feature": feature, **scores})
    return pd.DataFrame(rows, columns=["feature", "psi", "ks_statistic", "ks_pvalue", "js_divergence", "status"])


def load_prediction_events(log_path: Path = MONITORING_LOG_PATH) -> pd.DataFrame:
    if log_path == MONITORING_LOG_PATH:
        try:
            database_events = read_prediction_events()
            if not database_events.empty:
                return database_events
        except Exception:
            pass
    columns = ["timestamp", "path", "prediction", "probability", "churn_label"]
    if not log_path.exists():
        return pd.DataFrame(columns=columns)

    events: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                response = record.get("response", [])
                responses = response if isinstance(response, list) else [response]
                for item in responses:
                    if isinstance(item, dict) and "probability" in item:
                        events.append(
                            {
                                "timestamp": record.get("timestamp"),
                                "path": record.get("path", "unknown"),
                                "prediction": item.get("prediction"),
                                "probability": item.get("probability"),
                                "churn_label": item.get("churn_label"),
                            }
                        )
            except (json.JSONDecodeError, TypeError):
                continue

    frame = pd.DataFrame(events, columns=columns)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["probability"] = pd.to_numeric(frame["probability"], errors="coerce")
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    return frame.dropna(subset=["timestamp", "probability"])


def load_prediction_inputs(log_path: Path = MONITORING_LOG_PATH) -> pd.DataFrame:
    """Load raw customer inputs from API request logs for observed drift analysis."""
    if log_path == MONITORING_LOG_PATH:
        try:
            database_inputs = read_prediction_inputs()
            if not database_inputs.empty:
                return database_inputs
        except Exception:
            pass
    if not log_path.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                request = json.loads(line).get("request", {})
                rows.extend(request if isinstance(request, list) else [request])
            except (json.JSONDecodeError, TypeError):
                continue
    return pd.DataFrame(rows)


def build_monitoring_summary(events: pd.DataFrame) -> dict[str, pd.DataFrame | float | int]:
    if events.empty:
        empty = pd.DataFrame(columns=["date", "prediction_volume", "average_probability"])
        return {"trend": empty, "confidence": pd.DataFrame(columns=["confidence_band", "predictions"]), "volume": 0, "average_probability": 0.0}

    daily = events.assign(date=events["timestamp"].dt.date).groupby("date", as_index=False).agg(
        prediction_volume=("probability", "size"),
        average_probability=("probability", "mean"),
    )
    confidence = pd.cut(
        events["probability"].clip(0, 1),
        bins=[-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
        labels=["Very low", "Low", "Medium", "High", "Very high"],
    ).value_counts(sort=False).rename_axis("confidence_band").reset_index(name="predictions")
    return {
        "trend": daily,
        "confidence": confidence,
        "volume": int(len(events)),
        "average_probability": float(events["probability"].mean()),
    }


def simulate_drift(model: Any, training_frame: pd.DataFrame, sample_size: int = 250, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    source = training_frame.sample(n=sample_size, replace=True, random_state=seed).copy()
    source = source.rename(columns={"Online Security": "OnlineSecurity"})
    source["tenure"] = (source["tenure"] * rng.uniform(0.25, 0.7, len(source))).clip(0, 72).round().astype(int)
    source["MonthlyCharges"] = (source["MonthlyCharges"] * rng.uniform(1.05, 1.35, len(source))).clip(lower=0)
    source["TotalCharges"] = source["MonthlyCharges"] * source["tenure"]
    source["Contract"] = rng.choice(["Month-to-Month", "One Year", "Two Year"], len(source), p=[0.72, 0.2, 0.08])
    source["tenure_bucket"] = pd.cut(
        source["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
    )
    source["probability"] = model.predict_proba(source[model.feature_names_in_])[..., 1]
    source["cohort"] = "Simulated new data"
    return source[["probability", "cohort"]]


def write_feedback(prediction: int, probability: float, correct: bool, path: Path = FEEDBACK_LOG_PATH) -> None:
    if path == FEEDBACK_LOG_PATH:
        try:
            persist_feedback(prediction, probability, correct)
            return
        except Exception:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prediction": int(prediction),
        "probability": float(probability),
        "feedback": "correct" if correct else "incorrect",
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load_feedback(path: Path = FEEDBACK_LOG_PATH) -> pd.DataFrame:
    if path == FEEDBACK_LOG_PATH:
        try:
            database_feedback = read_feedback()
            if not database_feedback.empty:
                return database_feedback
        except Exception:
            pass
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "prediction", "probability", "feedback"])
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records, columns=["timestamp", "prediction", "probability", "feedback"])
