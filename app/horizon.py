from __future__ import annotations

from collections.abc import Iterable

import numpy as np

DEFAULT_HORIZONS_DAYS = (30, 60, 90)
BASELINE_HORIZON_DAYS = 90
BASELINE_HORIZON_MONTHS = 3.0


def estimate_horizon_probabilities(
    probability: float,
    horizons_days: Iterable[int] = DEFAULT_HORIZONS_DAYS,
    *,
    baseline_days: int = BASELINE_HORIZON_DAYS,
) -> dict[int, float]:
    """Estimate cumulative churn risk over several future horizons.

    The current dataset has an outcome label but no observation or churn dates, so
    it cannot provide separately supervised 30/60/90-day labels. We therefore
    interpret the calibrated model score as a 90-day baseline and spread its
    implied constant hazard across shorter horizons.
    """
    if baseline_days <= 0:
        raise ValueError("baseline_days must be positive")

    clipped_probability = float(np.clip(probability, 0.0, 1.0))
    estimates: dict[int, float] = {}
    for horizon in horizons_days:
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizons_days must contain positive integers")
        cumulative_probability = 1.0 - (1.0 - clipped_probability) ** (horizon / baseline_days)
        estimates[horizon] = round(float(np.clip(cumulative_probability, 0.0, 1.0)), 6)
    return estimates


def estimate_survival_metrics(
    probability: float,
    *,
    baseline_days: int = BASELINE_HORIZON_DAYS,
) -> dict[str, float]:
    """Return an exponential survival estimate from the calibrated baseline.

    This is a survival projection, not a fitted Cox model: the source dataset
    has no event or censoring timestamps. The calibrated 90-day risk defines a
    constant hazard, from which survival and median time-to-churn are derived.
    """
    if baseline_days <= 0:
        raise ValueError("baseline_days must be positive")

    clipped_probability = float(np.clip(probability, 0.0, 1.0))
    baseline_months = baseline_days / 30.0
    if clipped_probability >= 1.0:
        return {"hazard_per_month": float("inf"), "median_time_to_churn_months": 0.0}
    if clipped_probability <= 0.0:
        return {"hazard_per_month": 0.0, "median_time_to_churn_months": float("inf")}

    hazard_per_month = -np.log1p(-clipped_probability) / baseline_months
    median_time = np.log(2.0) / hazard_per_month
    return {
        "hazard_per_month": round(float(hazard_per_month), 6),
        "median_time_to_churn_months": round(float(median_time), 2),
    }
