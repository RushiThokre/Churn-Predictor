from app.business_metrics import calculate_customer_value


def test_customer_value_calculates_potential_revenue_and_loss():
    result = calculate_customer_value({"MonthlyCharges": 2499, "tenure": 0}, 0.91, planning_horizon=24)

    assert result["expected_remaining_months"] == 24
    assert result["potential_revenue"] == 59976
    assert result["revenue_at_risk"] == 54578.16


def test_customer_value_never_uses_less_than_one_year():
    result = calculate_customer_value({"MonthlyCharges": 100, "tenure": 60}, 1.5)

    assert result["expected_remaining_months"] == 12
    assert result["revenue_at_risk"] == 1200
