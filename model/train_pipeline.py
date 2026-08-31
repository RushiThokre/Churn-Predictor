from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco.csv"
MODEL_PATH = ROOT / "model" / "churn_model.pkl"


def build_pipeline() -> Pipeline:
    df = pd.read_csv(DATA_PATH)

    df = df.rename(
        columns={
            "Tenure in Months": "tenure",
            "Monthly Charge": "MonthlyCharges",
            "Total Charges": "TotalCharges",
            "Contract": "Contract",
            "Payment Method": "PaymentMethod",
            "Internet Service": "InternetService",
            "Tech Support": "TechSupport",
            "Premium Tech Support": "TechSupport",
            "Online Security": "OnlineSecurity",
            "Churn Label": "Churn",
        }
    )

    df["Churn"] = df["Churn"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).astype(int)
    df["tenure_bucket"] = pd.cut(df["tenure"], bins=[0, 12, 24, 48, 72], labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"])

    feature_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "PaymentMethod",
        "InternetService",
        "TechSupport",
        "OnlineSecurity",
        "tenure_bucket",
    ]

    X = df[feature_columns]
    y = df["Churn"]

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = ["Contract", "PaymentMethod", "InternetService", "TechSupport", "OnlineSecurity", "tenure_bucket"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                    eval_metric="logloss",
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = pipeline.predict(X_test)

    print("ROC-AUC:", roc_auc_score(y_test, probs))
    print("F1:", f1_score(y_test, preds))
    print(classification_report(y_test, preds))

    return pipeline


if __name__ == "__main__":
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("telco-churn")

    with mlflow.start_run(run_name="xgboost-baseline"):
        pipeline = build_pipeline()
        evaluation_frame = pd.read_csv(DATA_PATH).rename(
            columns={
                "Tenure in Months": "tenure",
                "Monthly Charge": "MonthlyCharges",
                "Total Charges": "TotalCharges",
                "Payment Method": "PaymentMethod",
                "Internet Service": "InternetService",
                "Premium Tech Support": "TechSupport",
                "Churn Label": "Churn",
            }
        )
        evaluation_frame["Churn"] = evaluation_frame["Churn"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).astype(int)
        evaluation_frame["tenure_bucket"] = pd.cut(
            evaluation_frame["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"],
        )
        feature_columns = [
            "tenure", "MonthlyCharges", "TotalCharges", "Contract", "PaymentMethod",
            "InternetService", "TechSupport", "OnlineSecurity", "tenure_bucket",
        ]
        _, test_features, _, test_target = train_test_split(
            evaluation_frame[feature_columns],
            evaluation_frame["Churn"],
            test_size=0.2,
            stratify=evaluation_frame["Churn"],
            random_state=42,
        )
        test_probabilities = pipeline.predict_proba(test_features)[:, 1]
        test_predictions = pipeline.predict(test_features)
        metrics = {
            "roc_auc": roc_auc_score(test_target, test_probabilities),
            "f1": f1_score(test_target, test_predictions),
        }
        mlflow.log_params({"model": "XGBClassifier", "random_state": 42, "test_size": 0.2})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        print(f"MLflow run: {mlflow.active_run().info.run_id}")
        print(f"Saved full pipeline to {MODEL_PATH}")
