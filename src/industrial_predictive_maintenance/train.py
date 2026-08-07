from pathlib import Path

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from industrial_predictive_maintenance.data import load_data
from industrial_predictive_maintenance.features import create_features
from industrial_predictive_maintenance.preprocessing import (
    split_features_target,
    split_data,
    build_preprocessor,
    fit_preprocessor,
    transform_data,
)
from industrial_predictive_maintenance.model import save_model


ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT_DIR / "data" / "raw" / "ai4i2020Raw.csv"
MODEL_PATH = ROOT_DIR / "models" / "model_artifacts.pkl"


def train() -> None:
    # Load data
    df = load_data(DATA_PATH)

    # Feature engineering
    df = create_features(df)

    # Split features and target
    X, y = split_features_target(df)

    # Train / validation / test split
    X_train, X_test, y_train, y_test = split_data(X, y)


    # Fit encoder on final training data
    encoder = build_preprocessor()

    fit_preprocessor( encoder, X_train)

    X_train = transform_data(encoder, X_train)

    # Final XGBoost model
    model = XGBClassifier(
        random_state=101,
        n_jobs=-1,
        subsample=1.0,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        colsample_bytree=1.0,
    )

    # Train final model
    model.fit(X_train, y_train)

    # Evaluate on the untouched test set
    X_test = transform_data(
        encoder,
        X_test,
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    print(
        f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}"
    )

    # Save model artifacts
    artifacts = {
        "preprocessor": encoder,
        "model": model,
        "feature_names": X_train.columns.to_list(),
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_model(
        artifacts,
        MODEL_PATH,
    )

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
