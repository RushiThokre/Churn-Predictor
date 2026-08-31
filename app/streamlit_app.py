from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it first with model/train_pipeline.py")
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("Customer Churn Predictor")
st.write("Enter customer details to estimate churn risk.")

with st.form("churn_form"):
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly charges", min_value=0.0, value=70.0, step=0.5)
    total_charges = st.number_input("Total charges", min_value=0.0, value=840.0, step=1.0)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    )
    internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech support", ["Yes", "No"])
    online_security = st.selectbox("Online security", ["Yes", "No"])
    submitted = st.form_submit_button("Predict churn")

if submitted:
    input_df = pd.DataFrame(
        [{
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract,
            "PaymentMethod": payment_method,
            "InternetService": internet_service,
            "TechSupport": tech_support,
            "OnlineSecurity": online_security,
        }]
    )

    input_df["tenure_bucket"] = pd.cut(
        input_df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
    )

    probability = model.predict_proba(input_df)[0][1]
    st.metric("Churn probability", f"{probability:.1%}")
    st.success("Will churn" if probability > 0.5 else "Will stay")
