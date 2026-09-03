from __future__ import annotations

from typing import Any


def calculate_customer_value(features: dict[str, Any], churn_probability: float, planning_horizon: int = 36) -> dict[str, float]:
    """Estimate customer value and expected revenue loss over a planning horizon."""
    monthly_charges = float(features.get("MonthlyCharges", 0) or 0)
    tenure = float(features.get("tenure", 0) or 0)
    expected_remaining_months = max(12.0, float(planning_horizon) - tenure)
    potential_revenue = monthly_charges * expected_remaining_months
    revenue_at_risk = potential_revenue * max(0.0, min(1.0, float(churn_probability)))
    return {
        "expected_remaining_months": expected_remaining_months,
        "potential_revenue": potential_revenue,
        "revenue_at_risk": revenue_at_risk,
    }
