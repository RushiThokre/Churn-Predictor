from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MONITORING_LOG_PATH = Path(__file__).resolve().parent.parent / "api" / "request_log.jsonl"
FEEDBACK_LOG_PATH = Path(__file__).resolve().parent.parent / "api" / "feedback_log.jsonl"


def load_prediction_events(log_path: Path = MONITORING_LOG_PATH) -> pd.DataFrame:
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
