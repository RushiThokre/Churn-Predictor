from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'churn.db'}")
_engine_kwargs: dict[str, Any] = {"pool_pre_ping": True}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    features: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), index=True)
    probability: Mapped[float] = mapped_column(Float)
    prediction: Mapped[int] = mapped_column(Integer)
    risk_level: Mapped[str] = mapped_column(String(16))
    revenue_at_risk: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str | None] = mapped_column(ForeignKey("customers.customer_id"), nullable=True)
    prediction: Mapped[int] = mapped_column(Integer)
    probability: Mapped[float] = mapped_column(Float)
    correct: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(128))
    artifact_path: Mapped[str] = mapped_column(String(512))
    decision_threshold: Mapped[float] = mapped_column(Float, default=0.5)
    is_champion: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class RetentionAction(Base):
    __tablename__ = "retention_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), index=True)
    reason: Mapped[str] = mapped_column(String(512))
    action: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


def init_db() -> None:
    Path(BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


def persist_prediction_request(payload: dict[str, Any] | list[dict[str, Any]], response: dict[str, Any] | list[dict[str, Any]]) -> None:
    requests = payload if isinstance(payload, list) else [payload]
    responses = response if isinstance(response, list) else [response]
    with SessionLocal.begin() as session:
        for index, (customer, result) in enumerate(zip(requests, responses)):
            customer_id = str(result.get("customer_id") or customer.get("customer_id") or f"CUST-{index + 1}")
            existing = session.get(Customer, customer_id)
            if existing is None:
                session.add(Customer(customer_id=customer_id, features=customer))
            else:
                existing.features = customer
                existing.updated_at = datetime.now(timezone.utc)
            session.add(Prediction(
                customer_id=customer_id,
                probability=float(result.get("churn_probability", result.get("probability", 0))),
                prediction=int(result.get("prediction", 0)),
                risk_level=str(result.get("risk_level", "LOW")),
                revenue_at_risk=float(result.get("revenue_at_risk", 0)),
            ))
            if result.get("recommended_action"):
                session.add(RetentionAction(customer_id=customer_id, reason=str(result.get("recommendation_reason", "")), action=str(result["recommended_action"])))


def persist_feedback(prediction: int, probability: float, correct: bool, customer_id: str | None = None) -> None:
    with SessionLocal.begin() as session:
        session.add(Feedback(customer_id=customer_id, prediction=int(prediction), probability=float(probability), correct=bool(correct)))


def persist_model_version(model_name: str, artifact_path: str, decision_threshold: float, is_champion: bool = True) -> None:
    with SessionLocal.begin() as session:
        if is_champion:
            for version in session.query(ModelVersion).filter(ModelVersion.is_champion.is_(True)).all():
                version.is_champion = False
        session.add(ModelVersion(model_name=model_name, artifact_path=artifact_path, decision_threshold=decision_threshold, is_champion=is_champion))


def read_prediction_events() -> pd.DataFrame:
    with SessionLocal() as session:
        rows = session.query(Prediction).order_by(Prediction.created_at).all()
    return pd.DataFrame([{"timestamp": row.created_at, "path": "database", "prediction": row.prediction, "probability": row.probability, "churn_label": "churn" if row.prediction else "stay"} for row in rows])


def read_prediction_inputs() -> pd.DataFrame:
    with SessionLocal() as session:
        rows = session.query(Customer).all()
    return pd.DataFrame([row.features for row in rows])


def read_feedback() -> pd.DataFrame:
    with SessionLocal() as session:
        rows = session.query(Feedback).order_by(Feedback.created_at).all()
    return pd.DataFrame([{"timestamp": row.created_at, "prediction": row.prediction, "probability": row.probability, "feedback": "correct" if row.correct else "incorrect"} for row in rows])
