# ChurnShield

AI-powered customer churn intelligence for the Telco customer dataset.

## What It Includes

- Multi-model training: Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost when installed.
- Stratified five-fold out-of-fold evaluation with class-imbalance handling.
- F1 threshold optimization and business-cost-aware champion selection.
- SHAP explanations for individual predictions.
- FastAPI prediction API with API-key authentication.
- SQLAlchemy persistence for customers, predictions, feedback, model versions, and retention actions.
- Streamlit prediction, batch analytics, feedback, and monitoring views.
- Revenue-at-risk estimates and rule-based retention recommendations.
- Observed-input monitoring with PSI, Kolmogorov-Smirnov, and Jensen-Shannon drift metrics plus retraining guidance.
- Pandera data validation for types, missing values, ranges, categories, duplicate IDs, and target leakage.
- Local MLflow experiment tracking and Docker support.

Predictions are expressed as cumulative churn risk over the next 30, 60, and 90
days for an active customer. The supplied historical dataset has a churn outcome
but no observation or churn dates, so the calibrated model score is treated as a
90-day baseline and converted to shorter horizons with a constant-hazard
assumption. A production model should replace this estimate with time-stamped
survival or horizon-specific labels.

## Architecture

```text
Telco CSV -> preprocessing -> model comparison -> champion artifact
                                      |                 |
                                  MLflow           FastAPI + Streamlit
                                                        |
                                  predictions -> monitoring -> feedback
```

## Run Locally

```bash
pip install -r requirements.txt
python -m model.train_pipeline
uvicorn api.main:app --reload
streamlit run app/streamlit_app.py
```

The API is available at `http://localhost:8000/docs` and the dashboard at `http://localhost:8501`.
Prediction requests require `X-API-Key`; the default development key is `demo-key`.

## API

```text
GET  /health
GET  /metrics
GET  /model-info
POST /predict
POST /predict/batch
GET  /api/v1/health
GET  /api/v1/metrics
GET  /api/v1/model-info
POST /api/v1/predict
POST /api/v1/batch-predict
```

Prediction routes accept either `X-API-Key` or an `Authorization: Bearer <JWT>` header and are rate limited per client. Responses retain the original `prediction`, `churn_label`, and `probability` fields and also include `churn_probability_30d`, `churn_probability_60d`, `churn_probability_90d`, `risk_level`, `revenue_at_risk`, and `recommended_action`. The legacy `probability` field is the 90-day baseline.

## Training Output

Running the training script saves `model/churn_model.pkl` and writes the ranked model metrics to `model/model_comparison.csv`. The local MLflow UI can be started with:

```bash
mlflow ui --backend-store-uri ./mlruns
```

## Docker

```bash
docker build -t churnshield .
docker run --rm -p 8000:8000 -p 8501:8501 -e API_KEY=change-me churnshield
```

For the PostgreSQL-backed stack:

```bash
docker compose up --build
```

The compose file provisions PostgreSQL and sets `DATABASE_URL` for the application. Local runs default to a SQLite file under `data/`; set `DATABASE_URL` to a PostgreSQL connection string in shared environments.

## Tests

```bash
pytest -q
```

The project uses the synthetic starter dataset in `data/telco.csv` so training and tests can run locally without external services.
