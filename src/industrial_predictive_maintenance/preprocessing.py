from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd


def split_features_target(
    df: pd.DataFrame,
    target: str = "machine_failure",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataset into features and target.
    """
    X = df.drop(columns=target)
    y = df[target]

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    random_state: int = 101,
):
    """
    Split the dataset into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)



def build_preprocessor():
    """Create the ordinal encoder."""
    return OrdinalEncoder(categories=[["L", "M", "H"]])

def fit_preprocessor(
    encoder: OrdinalEncoder,
    X_train: pd.DataFrame,
) -> None:
    """Fit the encoder on the 'type' column."""
    encoder.fit(X_train[["type"]])


def transform_data(
    encoder: OrdinalEncoder,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Encode the 'type' column."""
    X = X.copy()

    X["type"] = encoder.transform(X[["type"]])

    return X