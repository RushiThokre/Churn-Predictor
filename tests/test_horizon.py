import pytest

from app.horizon import estimate_horizon_probabilities, estimate_survival_metrics


def test_horizon_probabilities_are_cumulative_and_bounded():
    estimates = estimate_horizon_probabilities(0.81)

    assert estimates[30] < estimates[60] < estimates[90]
    assert estimates[90] == 0.81
    assert all(0 <= value <= 1 for value in estimates.values())


def test_horizon_probability_rejects_invalid_baseline():
    with pytest.raises(ValueError, match="baseline_days"):
        estimate_horizon_probabilities(0.5, baseline_days=0)


def test_survival_metrics_return_median_time_to_churn():
    metrics = estimate_survival_metrics(0.87)

    assert metrics["hazard_per_month"] > 0
    assert metrics["median_time_to_churn_months"] == 1.02
