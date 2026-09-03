from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

from app.data_validation import remove_target_leakage, validate_telco_data

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco.csv"
MODEL_PATH = ROOT / "model" / "churn_model.pkl"
COLUMN_RENAME_MAP = {
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


def build_pipeline() -> Pipeline:
    df = pd.read_csv(DATA_PATH).rename(columns=COLUMN_RENAME_MAP)

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


def _training_data() -> tuple[pd.DataFrame, pd.Series]:
    raw_frame = pd.read_csv(DATA_PATH)
    frame, leakage = remove_target_leakage(raw_frame)
    if leakage:
        print(f"Removed target leakage columns before training: {', '.join(leakage)}")
    frame = validate_telco_data(frame)
    frame["Churn"] = frame["Churn"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).astype(int)
    frame["tenure_bucket"] = pd.cut(frame["tenure"], bins=[0, 12, 24, 48, 72], labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"])
    features = ["tenure", "MonthlyCharges", "TotalCharges", "Contract", "PaymentMethod", "InternetService", "TechSupport", "OnlineSecurity", "tenure_bucket"]
    return frame[features], frame["Churn"]


class CalibratedPipeline:
    """Expose calibrated probabilities while retaining the base pipeline for SHAP."""

    def __init__(self, base_pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> None:
        self.base_pipeline = base_pipeline
        self.calibrated = CalibratedClassifierCV(estimator=base_pipeline, method="sigmoid", cv=3)
        self.calibrated.fit(X, y)
        self.decision_threshold = 0.5

    @property
    def named_steps(self):
        return self.base_pipeline.named_steps

    @property
    def feature_names_in_(self):
        return self.base_pipeline.feature_names_in_

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.calibrated.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.decision_threshold).astype(int)


def _candidate_pipelines(y_train: pd.Series, xgb_params: dict[str, object] | None = None) -> dict[str, Pipeline]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    numeric = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical = ["Contract", "PaymentMethod", "InternetService", "TechSupport", "OnlineSecurity", "tenure_bucket"]
    imbalance = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_settings: dict[str, object] = {
        "n_estimators": 250,
        "max_depth": 4,
        "learning_rate": 0.05,
    }
    xgb_settings.update(xgb_params or {})
    candidates: dict[str, object] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(**xgb_settings, scale_pos_weight=imbalance, eval_metric="logloss", random_state=42),
    }
    try:
        from lightgbm import LGBMClassifier
        candidates["LightGBM"] = LGBMClassifier(n_estimators=250, learning_rate=0.05, class_weight="balanced", verbosity=-1, random_state=42, n_jobs=1)
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
        candidates["CatBoost"] = CatBoostClassifier(iterations=250, depth=5, learning_rate=0.05, auto_class_weights="Balanced", verbose=False, random_seed=42, thread_count=1)
    except ImportError:
        pass
    return {
        name: Pipeline([
            ("preprocessor", ColumnTransformer([
                ("num", StandardScaler(), numeric),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ])),
            ("selector", SelectPercentile(score_func=f_classif, percentile=80)),
            ("classifier", classifier),
        ])
        for name, classifier in candidates.items()
    }


def _best_threshold(y_true: pd.Series, probabilities: pd.Series) -> float:
    thresholds = np.linspace(0.2, 0.8, 61)
    scores = [f1_score(y_true, probabilities >= threshold) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def _tune_xgboost(X: pd.DataFrame, y: pd.Series, splitter: StratifiedKFold) -> dict[str, object]:
    try:
        import optuna
    except ImportError:
        return {}

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 120, 300),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
        }
        model = _candidate_pipelines(y, params)["XGBoost"]
        probabilities = cross_val_predict(model, X, y, cv=splitter, method="predict_proba", n_jobs=1)[:, 1]
        return roc_auc_score(y, probabilities)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=5, show_progress_bar=False)
    return dict(study.best_params)


def train_champion() -> tuple[CalibratedPipeline, pd.DataFrame, float]:
    """Tune, compare, calibrate, and select a champion using out-of-fold predictions."""
    X, y = _training_data()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tuned_xgb = _tune_xgboost(X, y, splitter)
    rows = []
    trained: dict[str, Pipeline] = {}
    for name, candidate in _candidate_pipelines(y, tuned_xgb).items():
        try:
            probabilities = cross_val_predict(candidate, X, y, cv=splitter, method="predict_proba", n_jobs=1)[:, 1]
        except Exception as error:
            print(f"Skipping {name}: {error}")
            continue
        threshold = _best_threshold(y, pd.Series(probabilities))
        predictions = probabilities >= threshold
        business_cost = int(((y == 1) & ~predictions).sum() * 5 + ((y == 0) & predictions).sum())
        rows.append({"model": name, "roc_auc": roc_auc_score(y, probabilities), "f1": f1_score(y, predictions), "business_cost": business_cost, "threshold": threshold})
        candidate.fit(X, y)
        trained[name] = candidate
    results = pd.DataFrame(rows)
    results["score"] = results["roc_auc"] + results["f1"] - results["business_cost"] / max(len(y), 1)
    champion_name = str(results.sort_values("score", ascending=False).iloc[0]["model"])
    champion = CalibratedPipeline(trained[champion_name], X, y)
    champion.decision_threshold = float(results.loc[results["model"] == champion_name, "threshold"].iloc[0])
    return champion, results.sort_values("score", ascending=False).reset_index(drop=True), champion.decision_threshold


if __name__ == "__main__":
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    import mlflow

    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("telco-churn")

    with mlflow.start_run(run_name="champion-model-comparison"):
        pipeline, comparison, threshold = train_champion()
        champion = comparison.iloc[0]
        evaluation_frame = pd.read_csv(DATA_PATH).rename(columns=COLUMN_RENAME_MAP)
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
        metrics = {"roc_auc": float(champion["roc_auc"]), "f1": float(champion["f1"]), "business_cost": float(champion["business_cost"])}
        mlflow.log_params({"model": champion["model"], "cv": "StratifiedKFold(5)", "decision_threshold": threshold})
        mlflow.log_metrics(metrics)
        comparison.to_csv(ROOT / "model" / "model_comparison.csv", index=False)
        print(comparison.to_string(index=False))

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        print(f"MLflow run: {mlflow.active_run().info.run_id}")
        print(f"Saved full pipeline to {MODEL_PATH}")
