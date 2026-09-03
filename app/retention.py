from __future__ import annotations

from typing import Any


DEFAULT_ACTION = "Monitor account and send a retention check-in"


def recommend_retention_action(probability: float, features: dict[str, Any]) -> dict[str, str]:
    """Return a transparent rule-based retention recommendation."""
    monthly_charges = float(features.get("MonthlyCharges", 0) or 0)
    contract = str(features.get("Contract", "")).strip().lower()
    tech_support = str(features.get("TechSupport", "")).strip().lower()
    tenure = float(features.get("tenure", 0) or 0)

    if probability >= 0.80 and monthly_charges > 100:
        return {"reason": "High churn risk with high monthly charges", "recommended_action": "Offer a 15% billing discount"}
    if probability >= 0.75 and contract in {"month-to-month", "month-to-month"}:
        return {"reason": "High churn risk on a month-to-month contract", "recommended_action": "Offer an annual contract incentive"}
    if probability >= 0.70 and tech_support == "no":
        return {"reason": "High churn risk without technical support", "recommended_action": "Offer free technical support for 3 months"}
    if probability >= 0.70 and tenure <= 12:
        return {"reason": "High churn risk during the first year", "recommended_action": "Offer a new-customer loyalty incentive"}
    return {"reason": "No dominant intervention trigger", "recommended_action": DEFAULT_ACTION}
