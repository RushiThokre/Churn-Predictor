from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline


class CalibratedPipeline:
    """Expose calibrated 90-day baseline probabilities while retaining SHAP support."""

    def __init__(self, base_pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, horizons_days: tuple[int, ...] = (30, 60, 90)) -> None:
        self.base_pipeline = base_pipeline
        self.calibrated = CalibratedClassifierCV(estimator=base_pipeline, method="sigmoid", cv=3)
        self.calibrated.fit(X, y)
        self.decision_threshold = 0.5
        self.prediction_horizons_days = horizons_days
        self.target_definition = "Estimated probability an active customer churns within 30, 60, or 90 days"

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
