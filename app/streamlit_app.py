from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.explainability import (  # noqa: E402
    FEATURE_COLUMNS,
    explain_prediction,
    predict_batch,
    shap_waterfall,
    summarize_batch_predictions,
)
from app.business_metrics import calculate_customer_value  # noqa: E402
from app.horizon import estimate_horizon_probabilities, estimate_survival_metrics  # noqa: E402
from app.monitoring import (  # noqa: E402
    build_feature_drift_report,
    build_monitoring_summary,
    load_feedback,
    load_prediction_inputs,
    load_prediction_events,
    simulate_drift,
    write_feedback,
)
from app.retention import recommend_retention_action  # noqa: E402

MODEL_PATH = BASE_DIR / "model" / "churn_model.pkl"

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="app/assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
    }
    .brand {
        padding: 0.25rem 0 1.25rem;
    }
    .brand-title {
        color: #172554;
        font-size: 2rem;
        font-weight: 750;
        line-height: 1.1;
        letter-spacing: 0.01em;
    }
    .brand-subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-top: 0.45rem;
    }
    .model-status {
        border-left: 3px solid #22c55e;
        color: #166534;
        font-size: 0.82rem;
        font-weight: 750;
        line-height: 1.55;
        margin-top: 0.35rem;
        padding: 0.35rem 0 0.35rem 0.85rem;
        text-align: left;
    }
    .model-status small {
        color: #64748b;
        font-size: 0.76rem;
        font-weight: 500;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .status-safe {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .status-risk {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .input-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border-left: 5px solid #667eea;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it first with model/train_pipeline.py")
    return joblib.load(MODEL_PATH)


model = load_model()


@st.cache_data
def load_portfolio_predictions(_model) -> pd.DataFrame:
    """Score the bundled customer portfolio once per model instance."""
    source = pd.read_csv(BASE_DIR / "data" / "telco.csv")
    scored = predict_batch(_model, source)
    scored = scored.rename(columns={"Senior Citizen": "senior_citizen"})
    scored["risk_level"] = np.select(
        [scored["churn_probability"] >= 0.7, scored["churn_probability"] >= 0.4],
        ["High", "Medium"],
        default="Low",
    )
    recommendations = scored.apply(
        lambda row: recommend_retention_action(float(row["churn_probability"]), row.to_dict()),
        axis=1,
        result_type="expand",
    )
    recommendations.columns = ["recommendation_reason", "recommended_action"]
    scored = pd.concat([scored, recommendations], axis=1)
    value_columns = scored.apply(
        lambda row: calculate_customer_value(row.to_dict(), float(row["churn_probability"])),
        axis=1,
        result_type="expand",
    )
    scored = pd.concat([scored, value_columns], axis=1)
    return scored


def normalize_customer_input(row: dict) -> dict:
    contract_map = {
        "Month-to-month": "Month-to-Month",
        "Month-to-Month": "Month-to-Month",
        "One year": "One Year",
        "One Year": "One Year",
        "Two year": "Two Year",
        "Two Year": "Two Year",
    }
    payment_map = {
        "Electronic check": "Electronic Check",
        "Electronic Check": "Electronic Check",
        "Mailed check": "Mailed Check",
        "Mailed Check": "Mailed Check",
        "Bank transfer": "Bank Transfer",
        "Bank Transfer": "Bank Transfer",
        "Credit card": "Credit Card",
        "Credit Card": "Credit Card",
    }
    internet_map = {
        "Fiber optic": "Fiber Optic",
        "Fiber Optic": "Fiber Optic",
        "DSL": "DSL",
        "No": "No",
    }

    normalized = dict(row)
    normalized["Contract"] = contract_map.get(str(normalized.get("Contract", "")).strip(), str(normalized.get("Contract", "")).strip())
    normalized["PaymentMethod"] = payment_map.get(str(normalized.get("PaymentMethod", "")).strip(), str(normalized.get("PaymentMethod", "")).strip())
    normalized["InternetService"] = internet_map.get(str(normalized.get("InternetService", "")).strip(), str(normalized.get("InternetService", "")).strip())
    normalized["TechSupport"] = str(normalized.get("TechSupport", "")).strip()
    normalized["OnlineSecurity"] = str(normalized.get("OnlineSecurity", "")).strip()
    return normalized


# Header
col1, col2 = st.columns([4, 1])

with col1:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-title">🛡️ ChurnShield</div>
            <div class="brand-subtitle">Customer Churn Intelligence Platform</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="model-status">
            ● SYSTEM ONLINE<br>
            <small>XGBoost v1.4</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

portfolio_tab, single_tab, batch_tab, monitoring_tab = st.tabs(["Portfolio Overview", "Single Prediction", "Batch Upload & Analytics", "Model Monitoring"])

with portfolio_tab:
    st.markdown("### Customer risk dashboard")
    st.caption("Filter the scored portfolio to focus retention work on the customers and segments that matter most.")

    try:
        portfolio = load_portfolio_predictions(model)
        with st.sidebar:
            st.markdown("### Portfolio filters")
            contract_filter = st.multiselect("Contract type", sorted(portfolio["Contract"].dropna().unique()), placeholder="All contracts")
            tenure_filter = st.slider("Tenure (months)", 0, 72, (0, 72))
            internet_filter = st.multiselect("Internet service", sorted(portfolio["InternetService"].dropna().unique()), placeholder="All services")
            payment_filter = st.multiselect("Payment method", sorted(portfolio["PaymentMethod"].dropna().unique()), placeholder="All methods")
            senior_filter = st.selectbox("Senior citizen", ["All", "Yes", "No"])
            charge_min = float(portfolio["MonthlyCharges"].min())
            charge_max = float(portfolio["MonthlyCharges"].max())
            charge_filter = st.slider("Monthly charges", charge_min, charge_max, (charge_min, charge_max))
            risk_filter = st.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])

        filtered = portfolio[
            portfolio["tenure"].between(*tenure_filter)
            & portfolio["MonthlyCharges"].between(*charge_filter)
            & portfolio["risk_level"].isin(risk_filter)
        ].copy()
        if contract_filter:
            filtered = filtered[filtered["Contract"].isin(contract_filter)]
        if internet_filter:
            filtered = filtered[filtered["InternetService"].isin(internet_filter)]
        if payment_filter:
            filtered = filtered[filtered["PaymentMethod"].isin(payment_filter)]
        if senior_filter != "All":
            filtered = filtered[filtered["senior_citizen"].astype(str) == senior_filter]

        high_risk = int((filtered["risk_level"] == "High").sum())
        revenue_at_risk = float(filtered["revenue_at_risk"].sum())
        average_probability = float(filtered["churn_probability"].mean()) if not filtered.empty else 0.0
        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric("Customers", f"{len(filtered):,}")
        metric_2.metric("High risk", f"{high_risk:,}")
        metric_3.metric("Revenue at risk", f"${revenue_at_risk:,.0f}")
        metric_4.metric("Average churn probability", f"{average_probability:.1%}")

        chart_col, table_col = st.columns([0.9, 1.6])
        with chart_col:
            st.markdown("### Churn risk distribution")
            distribution = filtered["risk_level"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0).rename_axis("risk_level").reset_index(name="customers")
            risk_chart = px.bar(distribution, x="risk_level", y="customers", color="risk_level", color_discrete_map={"High": "#dc2626", "Medium": "#f59e0b", "Low": "#16a34a"})
            risk_chart.update_layout(showlegend=False, height=330, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(risk_chart, use_container_width=True)

        with table_col:
            st.markdown("### Top risk customers")
            top_risk = filtered.sort_values(["churn_probability", "revenue_at_risk"], ascending=False).head(10)
            top_risk = top_risk[["customer_id", "risk_level", "churn_probability", "MonthlyCharges", "revenue_at_risk", "Contract", "recommended_action"]].rename(columns={"customer_id": "Customer", "risk_level": "Risk", "churn_probability": "Churn probability", "MonthlyCharges": "Monthly revenue", "revenue_at_risk": "Revenue at risk", "recommended_action": "Recommended action"})
            top_risk["Churn probability"] = top_risk["Churn probability"].map(lambda value: f"{value:.1%}")
            top_risk["Monthly revenue"] = top_risk["Monthly revenue"].map(lambda value: f"${value:,.0f}")
            top_risk["Revenue at risk"] = top_risk["Revenue at risk"].map(lambda value: f"${value:,.0f}")
            st.dataframe(top_risk, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Unable to score the portfolio: {exc}")

with single_tab:
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown("### 📝 Customer Information")

    with st.form("churn_form", border=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            tenure = st.number_input("📅 Tenure (months)", min_value=0, max_value=72, value=12)
            contract = st.selectbox("📋 Contract Type", ["Month-to-month", "One year", "Two year"])
            internet_service = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])

        with col2:
            monthly_charges = st.number_input("💰 Monthly Charges ($)", min_value=0.0, value=70.0, step=0.5)
            payment_method = st.selectbox("💳 Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
            tech_support = st.selectbox("🔧 Tech Support", ["Yes", "No"])

        with col3:
            total_charges = st.number_input("💵 Total Charges ($)", min_value=0.0, value=840.0, step=1.0)
            online_security = st.selectbox("🔒 Online Security", ["Yes", "No"])
            st.empty()

        col1, col2, col3 = st.columns([1, 1, 2])
        with col3:
            submitted = st.form_submit_button("🚀 Predict Churn Risk", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        input_df = pd.DataFrame(
            [
                normalize_customer_input(
                    {
                        "tenure": tenure,
                        "MonthlyCharges": monthly_charges,
                        "TotalCharges": total_charges,
                        "Contract": contract,
                        "PaymentMethod": payment_method,
                        "InternetService": internet_service,
                        "TechSupport": tech_support,
                        "OnlineSecurity": online_security,
                    }
                )
            ]
        )
        input_df["tenure_bucket"] = pd.cut(
            input_df["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
        )

        probability = float(model.predict_proba(input_df)[0][1])
        horizon_probabilities = estimate_horizon_probabilities(probability)
        survival_metrics = estimate_survival_metrics(probability)
        explanation = explain_prediction(model, input_df).head(5)
        threshold = float(getattr(model, "decision_threshold", 0.5))
        st.session_state["latest_prediction"] = {"prediction": int(probability >= threshold), "probability": probability}

        st.markdown("### 📊 Prediction Results")

        st.caption("Estimated cumulative probability that this active customer churns within each future horizon.")
        horizon_col1, horizon_col2, horizon_col3 = st.columns(3)
        horizon_col1.metric("Next 30 days", f"{horizon_probabilities[30]:.1%}")
        horizon_col2.metric("Next 60 days", f"{horizon_probabilities[60]:.1%}")
        horizon_col3.metric("Next 90 days", f"{horizon_probabilities[90]:.1%}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">90-day Churn Probability</div>
                    <div class="metric-value">{horizon_probabilities[90]:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            if probability >= threshold:
                st.markdown(
                    """
                    <div class="status-risk">
                        ⚠️ HIGH RISK<br>
                        Customer is likely to churn
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:
                st.markdown(
                    """
                    <div class="status-safe">
                        ✅ LOW RISK<br>
                        Customer is likely to stay
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.metric("Estimated time to churn", f"~{survival_metrics['median_time_to_churn_months']:.1f} months", help="Median time-to-churn from an exponential survival projection using the 90-day calibrated risk.")

        st.markdown("### Risk Assessment")
        risk_level = "🔴 Very High" if probability > 0.8 else "🟠 High" if probability > 0.6 else "🟡 Medium" if probability > 0.4 else "🟢 Low"
        st.progress(float(probability), text=f"Risk Level: {risk_level}")
        recommendation = recommend_retention_action(probability, input_df.iloc[0].to_dict())
        st.info(f"**Recommended action:** {recommendation['recommended_action']}  \n_{recommendation['reason']}_")
        customer_value = calculate_customer_value(input_df.iloc[0].to_dict(), probability)
        value_col1, value_col2, value_col3 = st.columns(3)
        value_col1.metric("Expected remaining life", f"{customer_value['expected_remaining_months']:.0f} months")
        value_col2.metric("Potential revenue", f"${customer_value['potential_revenue']:,.0f}")
        value_col3.metric("Revenue at risk", f"${customer_value['revenue_at_risk']:,.0f}")

        st.markdown("### What-if retention simulation")
        st.caption("Change a few controllable customer conditions to estimate how a retention offer could affect churn risk.")
        with st.form("what_if_form", border=False):
            simulation_col1, simulation_col2, simulation_col3 = st.columns(3)
            with simulation_col1:
                simulated_contract = st.selectbox("Simulated contract", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(contract), key="simulated_contract")
            with simulation_col2:
                simulated_monthly_charges = st.number_input("Simulated monthly charges ($)", min_value=0.0, value=float(monthly_charges), step=0.5, key="simulated_monthly_charges")
            with simulation_col3:
                simulated_tenure = st.number_input("Simulated tenure (months)", min_value=0, max_value=72, value=int(tenure), key="simulated_tenure")
            simulated_tech_support = st.checkbox("Add tech support", value=tech_support == "Yes", key="simulated_tech_support")
            simulation_submitted = st.form_submit_button("Run simulation", use_container_width=True)

        if simulation_submitted:
            simulated_input = input_df.copy()
            simulated_input["Contract"] = normalize_customer_input({"Contract": simulated_contract})["Contract"]
            simulated_input["MonthlyCharges"] = simulated_monthly_charges
            simulated_input["tenure"] = simulated_tenure
            simulated_input["TechSupport"] = "Yes" if simulated_tech_support else "No"
            simulated_input["tenure_bucket"] = pd.cut(
                simulated_input["tenure"],
                bins=[0, 12, 24, 48, 72],
                labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
            )
            simulated_probability = float(model.predict_proba(simulated_input)[0][1])
            change_points = (probability - simulated_probability) * 100
            result_col1, result_col2, result_col3 = st.columns(3)
            result_col1.metric("Current probability", f"{probability:.1%}")
            result_col2.metric("Simulated probability", f"{simulated_probability:.1%}")
            result_col3.metric(
                "Potential churn reduction" if change_points >= 0 else "Potential churn increase",
                f"{abs(change_points):.1f} percentage points",
                delta=f"{change_points:+.1f} pp",
                delta_color="normal" if change_points >= 0 else "inverse",
            )

        explanation_col, chart_col = st.columns([1.15, 1])

        with explanation_col:
            st.markdown("### 🔍 Why this prediction?")
            st.caption("Positive values increase churn risk; percentages show each driver's share of total absolute SHAP impact.")
            if explanation.empty:
                st.info("No explanation data was generated for this prediction.")
            else:
                for _, row in explanation.iterrows():
                    sign = "+" if row["contribution"] >= 0 else "-"
                    st.write(f"- {row['label']} ({sign}{abs(row['share_pct']):.0f}%)")

        with chart_col:
            st.markdown("### 📈 Top Drivers")
            if explanation.empty:
                st.empty()
            else:
                chart_df = explanation[["label", "share_pct"]].rename(columns={"label": "feature", "share_pct": "impact"}).sort_values("impact", ascending=True)
                fig = px.bar(
                    chart_df,
                    x="impact",
                    y="feature",
                    orientation="h",
                    title="Top risk drivers",
                    color="impact",
                    color_continuous_scale="RdYlGn_r",
                )
                fig.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### SHAP waterfall")
        waterfall = shap_waterfall(model, input_df, max_display=8)
        if waterfall is not None:
            st.pyplot(waterfall, use_container_width=True)
            st.caption("The baseline is the model's average output. Red features push the prediction toward churn; blue features push it away.")

        with st.expander("📈 Customer Insights", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Tenure", f"{tenure} months", f"{tenure/12:.1f} years")
            col2.metric("Monthly Cost", f"${monthly_charges:.2f}")
            col3.metric("Total Paid", f"${total_charges:.2f}")
            col4.metric("Avg Monthly", f"${total_charges/max(tenure, 1):.2f}")

        feedback = st.columns(2)
        feedback[0].button(
            "Mark prediction correct",
            key="feedback_correct",
            on_click=write_feedback,
            kwargs={"prediction": st.session_state["latest_prediction"]["prediction"], "probability": probability, "correct": True},
        )
        feedback[1].button(
            "Mark prediction incorrect",
            key="feedback_incorrect",
            on_click=write_feedback,
            kwargs={"prediction": st.session_state["latest_prediction"]["prediction"], "probability": probability, "correct": False},
        )

with batch_tab:
    st.markdown("### 📦 Bulk customer upload")
    uploaded_file = st.file_uploader("Upload CSV with customer records", type=["csv"], help="Expected columns: tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod, InternetService, TechSupport, OnlineSecurity")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            predictions = predict_batch(model, batch_df)
            summary = summarize_batch_predictions(predictions)

            col1, col2, col3 = st.columns(3)
            col1.metric("Customers processed", len(predictions))
            col2.metric("Average churn probability", f"{summary['avg_probability']:.1%}")
            col3.metric("High-risk customers", summary["high_risk_count"])

            st.markdown("### Churn rate by contract type")
            contract_chart = px.bar(
                summary["contract_summary"],
                x="Contract",
                y="churn_rate_pct",
                color="churn_rate_pct",
                color_continuous_scale="Reds",
                title="Predicted churn rate by contract",
            )
            contract_chart.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(contract_chart, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Top 10 highest-risk customers")
                top_risk = summary["top_risk"].copy()
                top_risk["churn_probability"] = top_risk["churn_probability"].round(3)
                st.dataframe(top_risk, use_container_width=True, hide_index=True)

            with col2:
                st.markdown("### Avg churn probability by segment")
                segment_chart = px.bar(
                    summary["segment_summary"],
                    x="segment",
                    y="avg_probability_pct",
                    color="avg_probability_pct",
                    color_continuous_scale="Blues",
                    title="Average churn probability by segment",
                )
                segment_chart.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(segment_chart, use_container_width=True)

            st.markdown("### Batch predictions")
            display_df = predictions[["customer_id", "Contract", "InternetService", "TechSupport", "OnlineSecurity", "churn_probability_30d", "churn_probability_60d", "churn_probability_90d", "estimated_time_to_churn_months", "churn_label", "potential_revenue", "revenue_at_risk", "recommended_action"]].copy()
            for column in ["churn_probability_30d", "churn_probability_60d", "churn_probability_90d"]:
                display_df[column] = display_df[column].map(lambda value: f"{value:.1%}")
            st.dataframe(display_df.sort_values("churn_probability", ascending=False), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Unable to process the upload: {exc}")
            st.info("Use the expected column names: tenure, MonthlyCharges, TotalCharges, Contract, PaymentMethod, InternetService, TechSupport, OnlineSecurity.")
    else:
        st.info("Upload a CSV to generate predictions for all customers at once.")

with monitoring_tab:
    st.markdown("### Model monitoring")
    st.caption("Operational view of logged API predictions. Streamlit predictions can be labeled below for feedback review.")
    events = load_prediction_events()
    monitoring = build_monitoring_summary(events)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    feedback_frame = load_feedback()
    trend = monitoring.get("trend", pd.DataFrame(columns=["date", "prediction_volume", "average_probability"]))

    volume_delta = None
    risk_delta = None
    latest_trend_day = trend.iloc[-1] if len(trend) else None
    previous_trend_day = trend.iloc[-2] if len(trend) > 1 else None
    if latest_trend_day is not None:
        volume_delta = int(latest_trend_day["prediction_volume"])
    if latest_trend_day is not None and len(events) > 0:
        latest_date = pd.Timestamp(latest_trend_day["date"]).date()
        prior_events = events[events["timestamp"].dt.date < latest_date]
        if not prior_events.empty:
            risk_delta = float(monitoring["average_probability"] - prior_events["probability"].mean())

    with metric_col1:
        with st.container(border=True):
            st.markdown(
                '<div style="border-left: 4px solid #3b82f6; background: rgba(59, 130, 246, 0.07); border-radius: 0.75rem; padding: 0.25rem 0.75rem 0.5rem 0.75rem; margin: -0.25rem 0;">',
                unsafe_allow_html=True,
            )
            st.metric("Logged predictions", monitoring["volume"], delta=(f"{volume_delta:+d}" if volume_delta is not None else None), delta_color="normal")
            st.markdown("</div>", unsafe_allow_html=True)

    with metric_col2:
        with st.container(border=True):
            st.markdown(
                '<div style="border-left: 4px solid #f59e0b; background: rgba(245, 158, 11, 0.07); border-radius: 0.75rem; padding: 0.25rem 0.75rem 0.5rem 0.75rem; margin: -0.25rem 0;">',
                unsafe_allow_html=True,
            )
            st.metric(
                "Average churn probability",
                f"{monitoring['average_probability']:.1%}",
                delta=(f"{risk_delta:+.1%}" if risk_delta is not None else None),
                delta_color="normal",
            )
            st.markdown("</div>", unsafe_allow_html=True)

    with metric_col3:
        with st.container(border=True):
            st.markdown(
                '<div style="border-left: 4px solid #14b8a6; background: rgba(20, 184, 166, 0.07); border-radius: 0.75rem; padding: 0.25rem 0.75rem 0.5rem 0.75rem; margin: -0.25rem 0;">',
                unsafe_allow_html=True,
            )
            feedback_delta = None
            if "timestamp" in feedback_frame.columns and not feedback_frame.empty:
                timestamps = pd.to_datetime(feedback_frame["timestamp"], errors="coerce", utc=True).dropna()
                if not timestamps.empty:
                    day_counts = timestamps.dt.floor("D").value_counts().sort_index()
                    if len(day_counts) > 1:
                        feedback_delta = int(day_counts.iloc[-1] - day_counts.iloc[-2])
            st.metric("Feedback responses", len(feedback_frame), delta=(f"{feedback_delta:+d}" if feedback_delta is not None else None), delta_color="normal")
            st.markdown("</div>", unsafe_allow_html=True)

    if events.empty:
        st.info("No API prediction events are logged yet. Run a prediction request to populate this view.")
    else:
        st.markdown("### Prediction volume and probability trend")
        trend = monitoring["trend"]
        trend_chart = px.line(
            trend,
            x="date",
            y=["prediction_volume", "average_probability"],
            markers=True,
            title="Daily traffic and average probability",
        )
        trend_chart.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10), yaxis_title="Value")
        st.plotly_chart(trend_chart, use_container_width=True)

        confidence_chart = px.bar(
            monitoring["confidence"],
            x="confidence_band",
            y="predictions",
            color="confidence_band",
            title="Model confidence distribution",
        )
        confidence_chart.update_layout(height=330, showlegend=False, margin=dict(l=10, r=10, t=45, b=10))
        st.plotly_chart(confidence_chart, use_container_width=True)

    st.markdown("### Drift simulation")
    st.markdown("### Feature drift")
    st.caption("Observed API inputs are compared with the training population using PSI, KS, and Jensen-Shannon divergence.")
    observed_inputs = load_prediction_inputs()
    if observed_inputs.empty:
        st.info("No observed API feature inputs are available yet. Make a prediction request to populate this report.")
    else:
        reference_inputs = pd.read_csv(BASE_DIR / "data" / "telco.csv").rename(
            columns={
                "Tenure in Months": "tenure",
                "Monthly Charge": "MonthlyCharges",
                "Total Charges": "TotalCharges",
                "Payment Method": "PaymentMethod",
                "Internet Service": "InternetService",
                "Premium Tech Support": "TechSupport",
                "Online Security": "OnlineSecurity",
            }
        )
        drift_report = build_feature_drift_report(
            reference_inputs,
            observed_inputs,
            ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "PaymentMethod", "InternetService", "TechSupport", "OnlineSecurity"],
        )
        if drift_report.empty:
            st.warning("Observed requests do not contain enough comparable feature columns for drift analysis.")
        else:
            display_report = drift_report.copy()
            display_report["psi"] = display_report["psi"].map(lambda value: f"{value:.3f}")
            display_report["ks_statistic"] = display_report["ks_statistic"].map(lambda value: f"{value:.3f}")
            display_report["js_divergence"] = display_report["js_divergence"].map(lambda value: f"{value:.3f}")
            st.dataframe(display_report, use_container_width=True, hide_index=True)
            drifted_features = drift_report.loc[drift_report["status"] == "drift", "feature"].tolist()
            if drifted_features:
                st.error(f"Model retraining recommended. Significant drift detected in: {', '.join(drifted_features)}.")
            elif (drift_report["status"] == "warning").any():
                st.warning("Monitor feature drift closely; one or more features are in the warning range.")
            else:
                st.success("No significant feature drift detected in observed API inputs.")

    st.markdown("### Drift simulation")
    st.caption("Synthetic new data intentionally shifts tenure, pricing, and contract mix. This is a monitoring demo, not ground-truth performance measurement.")
    if st.button("Run synthetic drift comparison", key="run_drift"):
        training_frame = pd.read_csv(BASE_DIR / "data" / "telco.csv").rename(
            columns={
                "Tenure in Months": "tenure",
                "Monthly Charge": "MonthlyCharges",
                "Total Charges": "TotalCharges",
                "Payment Method": "PaymentMethod",
                "Internet Service": "InternetService",
                "Premium Tech Support": "TechSupport",
                "Online Security": "OnlineSecurity",
                "Churn Label": "Churn",
            }
        )
        training_frame["tenure_bucket"] = pd.cut(
            training_frame["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
        )
        baseline = pd.DataFrame({"probability": model.predict_proba(training_frame[model.feature_names_in_])[:, 1], "cohort": "Training distribution"})
        drifted = simulate_drift(model, training_frame)
        comparison = pd.concat([baseline.sample(n=min(250, len(baseline)), random_state=42), drifted], ignore_index=True)
        drift_chart = px.histogram(
            comparison,
            x="probability",
            color="cohort",
            nbins=20,
            barmode="overlay",
            opacity=0.72,
            title="Baseline vs simulated new-data predictions",
        )
        drift_chart.update_layout(height=360, margin=dict(l=10, r=10, t=45, b=10), xaxis_title="Churn probability")
        st.plotly_chart(drift_chart, use_container_width=True)
        baseline_mean = baseline["probability"].mean()
        drift_mean = drifted["probability"].mean()
        st.metric("Average probability shift", f"{drift_mean - baseline_mean:+.1%}")

    st.markdown("### Feedback loop")
    if "latest_prediction" not in st.session_state:
        st.info("Make a single prediction first, then label whether it was correct or incorrect.")
    else:
        latest = st.session_state["latest_prediction"]
        st.write(f"Latest prediction: **{'churn' if latest['prediction'] else 'stay'}** ({latest['probability']:.1%} probability)")
        feedback_col1, feedback_col2 = st.columns(2)
        feedback_col1.button("Correct", key="monitor_feedback_correct", on_click=write_feedback, kwargs={**latest, "correct": True})
        feedback_col2.button("Incorrect", key="monitor_feedback_incorrect", on_click=write_feedback, kwargs={**latest, "correct": False})
        if not feedback_frame.empty:
            st.dataframe(feedback_frame.tail(10), use_container_width=True, hide_index=True)
