# Churn Predictor

A small telco churn project scaffold with:

- `data/telco_churn.csv` for the dataset
- `notebooks/eda_and_training.ipynb` for exploration and model training
- `model/churn_model.pkl` for the serialized model artifact
- `api/main.py` for a prediction API
- `app/streamlit_app.py` for an interactive demo

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

## Notes

- The repository currently includes a synthetic starter dataset so the notebook can run end to end once Python is available.
- The API expects a trained model artifact at `model/churn_model.pkl`.
