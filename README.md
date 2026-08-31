<<<<<<< HEAD
# Churn Predictor

<p align="center"><img src="app/assets/churn_shield_logo.png" width="120"/></p>

A small telco churn project scaffold with:

- `data/telco_churn.csv` for the dataset
- `notebooks/eda_and_training.ipynb` for exploration and model training
- `model/churn_model.pkl` for the serialized model artifact
- `api/main.py` for a prediction API
- `app/streamlit_app.py` for an interactive demo
- `Dockerfile` for running the API and Streamlit dashboard together
- `.github/workflows/ci.yml` for automated tests and a training smoke check

## Project layout

```text
churn-predictor/
├── data/
│   └── telco_churn.csv
├── notebooks/
│   └── eda_and_training.ipynb
├── model/
│   └── churn_model.pkl
├── api/
│   └── main.py
├── app/
│   ├── assets/
│   │   └── churn_shield_logo.png
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

## How to use

1. Create a Python environment and install dependencies from `requirements.txt`.
2. Open `notebooks/eda_and_training.ipynb` to explore the data and train the model.
3. Save the trained pipeline to `model/churn_model.pkl`.
4. Start the API with `uvicorn api.main:app --reload`.
5. Start the Streamlit app with `streamlit run app/streamlit_app.py`.

## Docker

Build and run both services in one container:

```bash
docker build -t churn-predictor .
docker run --rm -p 8000:8000 -p 8501:8501 -e API_KEY=change-me churn-predictor
```

- API: `http://localhost:8000/docs`
- Dashboard: `http://localhost:8501`
- Prediction requests must send `X-API-Key: change-me`.

The dashboard's **Model Monitoring** tab reads API prediction events from `api/request_log.jsonl` and shows daily prediction volume, average churn probability, confidence bands, and a synthetic drift comparison. After making a single prediction, use the feedback controls to record a mock correct/incorrect label in `api/feedback_log.jsonl`.

## Experiment tracking

Training logs the XGBoost parameters, held-out ROC-AUC, F1 score, and model artifact to a local MLflow experiment:

```bash
python model/train_pipeline.py
mlflow ui --backend-store-uri ./mlruns
```

Open the MLflow UI at `http://localhost:5000` to compare runs. Each training execution also updates `model/churn_model.pkl`, which is the artifact used by the API and dashboard.

## Continuous integration

GitHub Actions runs on every push and pull request. It installs the requirements, runs the test suite, trains the model, and verifies that the model artifact is non-empty.

## Notes

- The repository currently includes a synthetic starter dataset so the notebook can run end to end once Python is available.
- The API expects a trained model artifact at `model/churn_model.pkl`.
=======
# Churn-Predictor
>>>>>>> 054228af9523f6169550ed04aaa41ed601c46efd
