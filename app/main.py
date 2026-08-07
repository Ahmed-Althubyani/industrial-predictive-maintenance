from pathlib import Path

import pandas as pd
from fastapi import FastAPI
import uvicorn

from app.schemas import MachineData, PredictionResponse
from industrial_predictive_maintenance.predict import Predictor


ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "model_artifacts.pkl"
)


app = FastAPI(
    title="Industrial Predictive Maintenance API",
    version="1.0.0",
)


predictor = Predictor(MODEL_PATH)


@app.get("/")
def root():
    return {
        "message": "Industrial Predictive Maintenance API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(data: MachineData) -> PredictionResponse:

    df = pd.DataFrame([data.model_dump()])

    prediction = predictor.predict(df)[0]

    probability = predictor.predict_probability(df)[0]

    return PredictionResponse(
        prediction=prediction,
        failure_probability=probability,
    )



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)