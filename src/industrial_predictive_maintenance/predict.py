from pathlib import Path

import pandas as pd

from industrial_predictive_maintenance.model import load_model
from industrial_predictive_maintenance.features import create_features
from industrial_predictive_maintenance.preprocessing import transform_data


class Predictor:
    """Predict machine failure."""

    def __init__(self, model_path: Path):
        artifacts = load_model(model_path)

        self.model = artifacts["model"]
        self.encoder = artifacts["preprocessor"]
        self.feature_names = artifacts["feature_names"]
        self.threshold = artifacts.get("threshold", 0.5)

    def predict_probability( self, data: pd.DataFrame) -> list[float]:
        data = create_features(data)
        data = transform_data(self.encoder, data)

        probabilities = self.model.predict_proba(data)[:, 1]

        return probabilities.tolist()


    def predict(self, data: pd.DataFrame) -> list[int]:
        data = create_features(data)
        data = transform_data(self.encoder, data)

        predictions = self.model.predict(data)

        return predictions.tolist()