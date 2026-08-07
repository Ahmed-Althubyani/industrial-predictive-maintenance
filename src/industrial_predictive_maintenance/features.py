import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features.
    """

    df = df.copy()

    df["temperature_difference"] = (
        df["process_temperature"] -
        df["air_temperature"]
    )

    df["power"] = (
        df["torque"] *
        df["rotational_speed"]
    )

    return df