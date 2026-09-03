from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError


REQUIRED_COLUMNS = {
    "Customer ID": "customer_id",
    "Tenure in Months": "tenure",
    "Monthly Charge": "MonthlyCharges",
    "Total Charges": "TotalCharges",
    "Contract": "Contract",
    "Payment Method": "PaymentMethod",
    "Internet Service": "InternetService",
    "Premium Tech Support": "TechSupport",
    "Online Security": "OnlineSecurity",
    "Churn Label": "Churn",
}
TARGET_LEAKAGE_COLUMNS = {
    "Customer Status",
    "Churn Score",
    "CLTV",
    "Churn Category",
    "Churn Reason",
}
ALLOWED_CATEGORIES = {
    "Contract": {"Month-to-Month", "One Year", "Two Year"},
    "PaymentMethod": {"Bank Transfer", "Credit Card", "Electronic Check", "Mailed Check", "Bank Withdrawal"},
    "InternetService": {"Yes", "DSL", "Fiber Optic", "No"},
    "TechSupport": {"Yes", "No"},
    "OnlineSecurity": {"Yes", "No"},
    "Churn": {"Yes", "No"},
}


def find_target_leakage(frame: pd.DataFrame) -> list[str]:
    """Return columns that are populated after the churn outcome is known."""
    return sorted(TARGET_LEAKAGE_COLUMNS.intersection(frame.columns))


def _canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        **REQUIRED_COLUMNS,
        "Tech Support": "TechSupport",
        "Internet Service": "InternetService",
        "Payment Method": "PaymentMethod",
        "CustomerID": "customer_id",
    }
    return frame.rename(columns=aliases).copy()


def validate_telco_data(frame: pd.DataFrame, *, allow_target_leakage: bool = False) -> pd.DataFrame:
    """Validate raw telco data and return canonicalized columns.

    Leakage columns are rejected by default because they are unavailable at prediction time.
    """
    canonical = _canonicalize_columns(frame)
    leakage = find_target_leakage(canonical)
    if leakage and not allow_target_leakage:
        raise ValueError(f"Target leakage columns found: {', '.join(leakage)}")

    schema = pa.DataFrameSchema(
        {
            "customer_id": pa.Column(str, nullable=False, unique=True),
            "tenure": pa.Column(float, checks=pa.Check.in_range(0, 72), nullable=False, coerce=True),
            "MonthlyCharges": pa.Column(float, checks=pa.Check.in_range(0, 500), nullable=False, coerce=True),
            "TotalCharges": pa.Column(float, checks=pa.Check.in_range(0, 100000), nullable=False, coerce=True),
            **{
                column: pa.Column(str, checks=pa.Check.isin(sorted(values)), nullable=False)
                for column, values in ALLOWED_CATEGORIES.items()
                if column != "Churn"
            },
            "Churn": pa.Column(str, checks=pa.Check.isin(["Yes", "No"]), nullable=False),
        },
        strict=False,
        coerce=False,
    )
    try:
        return schema.validate(canonical)
    except SchemaError as error:
        raise ValueError(f"Telco data validation failed: {error.failure_cases}") from error


def remove_target_leakage(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop known post-outcome fields before model training and report what was removed."""
    canonical = _canonicalize_columns(frame)
    leakage = find_target_leakage(canonical)
    return canonical.drop(columns=leakage, errors="ignore"), leakage
