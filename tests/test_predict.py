from pathlib import Path

import pandas as pd

from industrial_predictive_maintenance.predict import Predictor
from industrial_predictive_maintenance.data import load_data


ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "model_artifacts.pkl"
)

DATA_PATH = (
    ROOT_DIR
    / "data"
    / "raw"
    / "ai4i2020Raw.csv"
)

df = load_data(DATA_PATH)

sample = df.drop(columns="machine_failure").iloc[[0]]

predictor = Predictor(MODEL_PATH)

prediction = predictor.predict(sample)
probability = predictor.predict_probability(sample)

print("Prediction:", prediction[0])
print("Failure probability:", probability[0])