from pathlib import Path

import joblib


def save_model(artifacts: dict, path: str | Path) -> None:
    """
    Save model artifacts to disk.
    """
    joblib.dump(artifacts, path)


def load_model(path: str | Path) -> dict:
    """
    Load model artifacts from disk.
    """
    return joblib.load(path)