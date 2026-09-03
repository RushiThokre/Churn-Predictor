from app.retention import recommend_retention_action


def test_high_bill_rule_has_priority():
    result = recommend_retention_action(0.91, {"MonthlyCharges": 120, "Contract": "Month-to-month"})
    assert result["recommended_action"] == "Offer a 15% billing discount"


def test_contract_rule_recommends_annual_plan():
    result = recommend_retention_action(0.86, {"MonthlyCharges": 80, "Contract": "Month-to-month"})
    assert result["recommended_action"] == "Offer an annual contract incentive"


def test_low_tenure_rule_is_available():
    result = recommend_retention_action(0.78, {"MonthlyCharges": 60, "Contract": "Two Year", "TechSupport": "Yes", "tenure": 4})
    assert result["recommended_action"] == "Offer a new-customer loyalty incentive"
