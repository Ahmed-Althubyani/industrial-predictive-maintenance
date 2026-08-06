from pathlib import Path

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load a dataset from the given path."""

    return pd.read_csv(Path(path))