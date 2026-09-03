import pandas as pd
import pytest

from app.data_validation import remove_target_leakage, validate_telco_data


@pytest.fixture
def valid_frame() -> pd.DataFrame:
    return pd.DataFrame([{
        "Customer ID": "C-001",
        "Tenure in Months": 12,
        "Monthly Charge": 70.0,
        "Total Charges": 840.0,
        "Contract": "Month-to-Month",
        "Payment Method": "Electronic Check",
        "Internet Service": "Fiber Optic",
        "Premium Tech Support": "No",
        "Online Security": "No",
        "Churn Label": "No",
    }])


def test_valid_data_is_canonicalized(valid_frame):
    validated = validate_telco_data(valid_frame)
    assert {"customer_id", "tenure", "MonthlyCharges", "TotalCharges", "Churn"}.issubset(validated.columns)


def test_duplicate_customer_ids_are_rejected(valid_frame):
    duplicate = pd.concat([valid_frame, valid_frame], ignore_index=True)
    with pytest.raises(ValueError, match="validation failed"):
        validate_telco_data(duplicate)


def test_range_violations_are_rejected(valid_frame):
    valid_frame.loc[0, "Tenure in Months"] = 90
    with pytest.raises(ValueError, match="validation failed"):
        validate_telco_data(valid_frame)


def test_unexpected_categories_are_rejected(valid_frame):
    valid_frame.loc[0, "Contract"] = "Lifetime"
    with pytest.raises(ValueError, match="validation failed"):
        validate_telco_data(valid_frame)


def test_target_leakage_is_rejected_and_can_be_removed(valid_frame):
    valid_frame["Churn Reason"] = "Price"
    with pytest.raises(ValueError, match="Target leakage"):
        validate_telco_data(valid_frame)

    cleaned, leakage = remove_target_leakage(valid_frame)
    assert leakage == ["Churn Reason"]
    assert "Churn Reason" not in cleaned.columns
