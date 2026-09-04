from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from app.horizon import estimate_horizon_probabilities, estimate_survival_metrics

FEATURE_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract",
    "PaymentMethod",
    "InternetService",
    "TechSupport",
    "OnlineSecurity",
]


def _normalize_contract(value: object) -> str:
    cleaned = str(value).strip() if value is not None else ""
    mapping = {
        "month-to-month": "Month-to-Month",
        "month to month": "Month-to-Month",
        "month-to-month ": "Month-to-Month",
        "one year": "One Year",
        "one-year": "One Year",
        "two year": "Two Year",
        "two-year": "Two Year",
    }
    return mapping.get(cleaned.lower(), cleaned)


def _normalize_payment(value: object) -> str:
    cleaned = str(value).strip() if value is not None else ""
    mapping = {
        "electronic check": "Electronic Check",
        "mailed check": "Mailed Check",
        "bank transfer": "Bank Transfer",
        "credit card": "Credit Card",
    }
    return mapping.get(cleaned.lower(), cleaned)


def _normalize_internet(value: object) -> str:
    cleaned = str(value).strip() if value is not None else ""
    mapping = {
        "fiber optic": "Fiber Optic",
        "fiber": "Fiber Optic",
        "dsl": "DSL",
        "no": "No",
    }
    return mapping.get(cleaned.lower(), cleaned)


def _normalize_yes_no(value: object) -> str:
    cleaned = str(value).strip() if value is not None else ""
    mapping = {"yes": "Yes", "no": "No", "y": "Yes", "n": "No"}
    return mapping.get(cleaned.lower(), cleaned)


def _prepare_input_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    rename_map = {
        "Tenure in Months": "tenure",
        "Monthly Charge": "MonthlyCharges",
        "Total Charges": "TotalCharges",
        "Contract": "Contract",
        "Payment Method": "PaymentMethod",
        "Internet Service": "InternetService",
        "Tech Support": "TechSupport",
        "Premium Tech Support": "TechSupport",
        "Online Security": "OnlineSecurity",
        "Customer ID": "customer_id",
        "CustomerID": "customer_id",
    }
    df = df.rename(columns=rename_map)

    if "customer_id" not in df.columns:
        df["customer_id"] = [f"CUST-{index + 1}" for index in range(len(df))]

    required_columns = FEATURE_COLUMNS
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing required feature columns: {missing_columns}")

    df["Contract"] = df["Contract"].apply(_normalize_contract)
    df["PaymentMethod"] = df["PaymentMethod"].apply(_normalize_payment)
    df["InternetService"] = df["InternetService"].apply(_normalize_internet)
    df["TechSupport"] = df["TechSupport"].apply(_normalize_yes_no)
    df["OnlineSecurity"] = df["OnlineSecurity"].apply(_normalize_yes_no)

    if "tenure_bucket" not in df.columns:
        df["tenure_bucket"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
        )

    return df


def predict_batch(model, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return preview predictions for a batch of customer records."""
    df = _prepare_input_features(raw_df)
    feature_frame = df[FEATURE_COLUMNS + ["tenure_bucket"]]
    probabilities = model.predict_proba(feature_frame)[0:, 1]

    predictions = df.copy()
    predictions["churn_probability"] = probabilities
    horizon_values = [estimate_horizon_probabilities(float(probability)) for probability in probabilities]
    predictions["churn_probability_30d"] = [values[30] for values in horizon_values]
    predictions["churn_probability_60d"] = [values[60] for values in horizon_values]
    predictions["churn_probability_90d"] = [values[90] for values in horizon_values]
    predictions["estimated_time_to_churn_months"] = [
        estimate_survival_metrics(float(probability))["median_time_to_churn_months"]
        for probability in probabilities
    ]
    threshold = float(getattr(model, "decision_threshold", 0.5))
    predictions["churn_label"] = predictions["churn_probability"].apply(lambda prob: "churn" if prob >= threshold else "stay")
    return predictions


def summarize_batch_predictions(predictions: pd.DataFrame) -> dict[str, pd.DataFrame | float | int]:
    if predictions.empty:
        return {"contract_summary": pd.DataFrame(columns=["Contract", "churn_rate_pct"]), "top_risk": pd.DataFrame(columns=["customer_id", "Contract", "churn_probability"]), "segment_summary": pd.DataFrame(columns=["segment", "avg_probability_pct"]), "avg_probability": 0.0, "high_risk_count": 0}

    contract_summary = (
        predictions.groupby("Contract", dropna=False)["churn_label"].apply(lambda values: (values == "churn").mean() * 100).reset_index(name="churn_rate_pct")
    )

    top_risk = predictions[["customer_id", "Contract", "churn_probability"]].sort_values("churn_probability", ascending=False).head(10).copy()
    top_risk = top_risk.reset_index(drop=True)

    segment_summary = (
        predictions.assign(segment=predictions["Contract"])
        .groupby("segment", dropna=False)["churn_probability"].mean().mul(100).reset_index(name="avg_probability_pct")
    )

    return {
        "contract_summary": contract_summary,
        "top_risk": top_risk,
        "segment_summary": segment_summary,
        "avg_probability": float(predictions["churn_probability"].mean()),
        "high_risk_count": int((predictions["churn_probability"] >= 0.5).sum()),
    }


def _decode_encoded_feature(feature_name: str) -> str:
    cleaned = feature_name.replace("num__", "").replace("cat__", "")

    if cleaned.startswith("Contract_"):
        return "Contract type"
    if cleaned.startswith("PaymentMethod_"):
        return "Payment method"
    if cleaned.startswith("InternetService_"):
        return "Internet service"
    if cleaned.startswith("TechSupport_"):
        return "Tech support"
    if cleaned.startswith("OnlineSecurity_"):
        return "Online security"
    if cleaned.startswith("tenure_bucket_"):
        if cleaned.endswith("0-1yr"):
            return "Low tenure"
        if cleaned.endswith("1-2yr"):
            return "Medium tenure"
        if cleaned.endswith("2-4yr"):
            return "Established tenure"
        return "Long tenure"
    if cleaned == "tenure":
        return "Tenure"
    if cleaned == "MonthlyCharges":
        return "Monthly charges"
    if cleaned == "TotalCharges":
        return "Total charges"
    return cleaned.replace("_", " ")


def _build_shap_explanation(model, raw_df: pd.DataFrame) -> shap.Explanation | None:
    if not hasattr(model, "named_steps"):
        return None

    preprocessor = model.named_steps.get("preprocessor")
    selector = model.named_steps.get("selector")
    classifier = model.named_steps.get("classifier")
    if preprocessor is None or classifier is None:
        return None

    input_df = _prepare_input_features(raw_df)
    transformed = preprocessor.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = np.asarray(preprocessor.get_feature_names_out())
    if selector is not None:
        transformed = selector.transform(transformed)
        feature_names = feature_names[selector.get_support()]

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    values = np.asarray(shap_values)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    elif values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    elif values.ndim != 1:
        values = values.reshape(-1)

    if len(values) < len(feature_names):
        values = np.pad(values, (0, len(feature_names) - len(values)), constant_values=0)

    expected_value = np.asarray(explainer.expected_value)
    base_value = float(expected_value[1] if expected_value.ndim else expected_value)
    return shap.Explanation(
        values=values[:len(feature_names)],
        base_values=base_value,
        data=transformed[0],
        feature_names=[_decode_encoded_feature(name) for name in feature_names],
    )


def explain_prediction(model, raw_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Return the top signed churn drivers for a prediction using SHAP."""
    explanation = _build_shap_explanation(model, raw_df)
    if explanation is None:
        return pd.DataFrame(columns=["feature", "label", "contribution", "share_pct"])

    contribution_df = pd.DataFrame({
        "feature": explanation.feature_names,
        "contribution": explanation.values,
    })
    contribution_df["label"] = contribution_df["feature"].map(_decode_encoded_feature)

    positive = contribution_df[contribution_df["contribution"] > 0].copy()
    if positive.empty:
        positive = contribution_df.copy()

    positive["abs_contribution"] = positive["contribution"].abs()
    if positive["abs_contribution"].sum() == 0:
        return pd.DataFrame(columns=["feature", "label", "contribution", "share_pct"])

    positive["share_pct"] = positive["abs_contribution"] / positive["abs_contribution"].sum() * 100
    positive = positive.sort_values("contribution", ascending=False).reset_index(drop=True)

    return positive.head(top_n)[["feature", "label", "contribution", "share_pct"]].copy()


def _driver_label(feature_name: str) -> str:
    cleaned = feature_name.replace("num__", "").replace("cat__", "")
    if cleaned == "MonthlyCharges":
        return "High monthly charges"
    if cleaned == "TotalCharges":
        return "High total charges"
    if cleaned == "tenure":
        return "Low tenure"
    if cleaned.startswith("Contract_"):
        return f"{cleaned.removeprefix('Contract_')} contract"
    if cleaned.startswith("PaymentMethod_"):
        return f"{cleaned.removeprefix('PaymentMethod_')} payment"
    if cleaned.startswith("TechSupport_No"):
        return "No tech support"
    if cleaned.startswith("OnlineSecurity_No"):
        return "No online security"
    return _decode_encoded_feature(feature_name)


def summarize_churn_drivers(model, raw_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Aggregate mean absolute SHAP impact across a portfolio."""
    if raw_df.empty or not hasattr(model, "named_steps"):
        return pd.DataFrame(columns=["label", "impact", "share_pct"])

    preprocessor = model.named_steps.get("preprocessor")
    selector = model.named_steps.get("selector")
    classifier = model.named_steps.get("classifier")
    if preprocessor is None or classifier is None:
        return pd.DataFrame(columns=["label", "impact", "share_pct"])

    input_df = _prepare_input_features(raw_df)
    transformed = preprocessor.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = np.asarray(preprocessor.get_feature_names_out())
    if selector is not None:
        transformed = selector.transform(transformed)
        feature_names = feature_names[selector.get_support()]

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    values = np.asarray(shap_values)
    if values.ndim != 2:
        return pd.DataFrame(columns=["label", "impact", "share_pct"])

    summary = pd.DataFrame({
        "label": [_driver_label(name) for name in feature_names],
        "impact": np.abs(values).mean(axis=0),
    })
    summary = summary.groupby("label", as_index=False)["impact"].sum()
    total_impact = float(summary["impact"].sum())
    if total_impact == 0:
        return pd.DataFrame(columns=["label", "impact", "share_pct"])
    summary["share_pct"] = summary["impact"] / total_impact * 100
    return summary.sort_values("impact", ascending=False).head(top_n).reset_index(drop=True)


def shap_waterfall(model, raw_df: pd.DataFrame, max_display: int = 8):
    """Return a matplotlib SHAP waterfall figure for the first input row."""
    explanation = _build_shap_explanation(model, raw_df)
    if explanation is None:
        return None
    plt.close("all")
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    return plt.gcf()
