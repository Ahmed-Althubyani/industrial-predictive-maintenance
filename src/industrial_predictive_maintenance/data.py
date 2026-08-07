from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Load and clean the predictive maintenance dataset."""


    df = pd.read_csv(path)

    # Drop unused columns
    df = df.drop(columns=["UDI",
         "Product ID", "TWF",
         "HDF",
         "PWF",
         "OSF",
         "RNF",])

    # Standardize column names
    df.columns = df.columns.str.lower().str.replace(r'\s*\[.*?\]$', '', regex=True).str.replace(" ", "_")

    return df