
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.schema import FEATURE_COLUMNS
from src.features.encoder import CATEGORICAL_COLUMNS
from src.features.scaler import NUMERICAL_COLUMNS


# ============================================================
# PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = ROOT_DIR / "models" / "artifacts"

MODEL_PATH = ARTIFACTS_DIR / "xgboost_pipeline.joblib"
ENCODER_PATH = ARTIFACTS_DIR / "xgboost_label_encoder.joblib"


# ============================================================
# LOAD MODEL
# ============================================================

def load_artifacts():
    """Load the trained model and label encoder."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    if not ENCODER_PATH.exists():
        raise FileNotFoundError(
            f"Label encoder not found: {ENCODER_PATH}"
        )

    try:
        loaded_model = joblib.load(MODEL_PATH)
        loaded_encoder = joblib.load(ENCODER_PATH)

    except Exception as error:
        raise RuntimeError(
            f"Failed to load model artifacts: {error}"
        ) from error

    return loaded_model, loaded_encoder


# Load once when the module is imported
model, label_encoder = load_artifacts()


# ============================================================
# INPUT VALIDATION
# ============================================================

def prepare_input(network_traffic):
    """
    Validate and prepare network traffic features.

    Accepts:
        - Dictionary: one network connection.
        - DataFrame: one or more network connections.

    Returns:
        DataFrame containing the 41 required features.
    """

    if isinstance(network_traffic, dict):

        X = pd.DataFrame([network_traffic])

    elif isinstance(network_traffic, pd.DataFrame):

        X = network_traffic.copy()

    else:

        raise TypeError(
            "Input must be a dictionary or pandas DataFrame."
        )

    # Reject empty input
    if X.empty:
        raise ValueError(
            "Input DataFrame is empty."
        )

    # Reject duplicate columns
    if X.columns.duplicated().any():
        raise ValueError(
            "Input contains duplicate column names."
        )

    # Check required features
    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in X.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required features: {missing_columns}"
        )

    # Keep only required features in official order
    X = X[FEATURE_COLUMNS].copy()

    # Reject missing values
    if X.isnull().any().any():

        missing_values = X.columns[
            X.isnull().any()
        ].tolist()

        raise ValueError(
            f"Input contains missing values in: {missing_values}"
        )

    # Validate categorical features
    for column in CATEGORICAL_COLUMNS:

        invalid_values = X[column].map(
            lambda value: isinstance(value, str)
            and value.strip() == ""
        )

        if invalid_values.any():
            raise ValueError(
                f"Empty categorical values found in: {column}"
            )

    # Validate numerical features
    for column in NUMERICAL_COLUMNS:

        try:
            X[column] = pd.to_numeric(
                X[column],
                errors="raise"
            )

        except (ValueError, TypeError) as error:

            raise ValueError(
                f"Column '{column}' must contain numeric values."
            ) from error

        if not np.isfinite(
            X[column].to_numpy(dtype=float)
        ).all():

            raise ValueError(
                f"Column '{column}' contains infinite or invalid values."
            )

    return X


# ============================================================
# BATCH PREDICTION
# ============================================================

def predict_batch(network_traffic):
    """
    Predict attack classes for one or more network connections.

    Returns:
        List of dictionaries containing:
            - predicted_attack
            - confidence
            - probabilities
    """

    X = prepare_input(network_traffic)

    # Model predictions
    encoded_predictions = np.asarray(
        model.predict(X),
        dtype=int
    )

    probabilities = model.predict_proba(X)

    # Validate prediction dimensions
    if len(encoded_predictions) != len(X):
        raise RuntimeError(
            "Prediction count does not match input count."
        )

    if probabilities.shape != (
        len(X),
        len(label_encoder.classes_)
    ):
        raise RuntimeError(
            "Prediction probabilities have an unexpected shape."
        )

    # Convert encoded predictions to class names
    predicted_classes = label_encoder.inverse_transform(
        encoded_predictions
    )

    results = []

    for index, predicted_class in enumerate(predicted_classes):

        class_probabilities = {
            str(class_name): float(probability)
            for class_name, probability in zip(
                label_encoder.classes_,
                probabilities[index]
            )
        }

        predicted_index = encoded_predictions[index]

        confidence = float(
            probabilities[index, predicted_index]
        )

        results.append({
            "predicted_attack": str(predicted_class),
            "confidence": round(confidence, 4),
            "probabilities": class_probabilities
        })

    return results


# ============================================================
# SINGLE PREDICTION
# ============================================================

def predict_attack(network_traffic):
    """
    Predict the attack class for exactly one connection.

    Returns:
        Dictionary containing prediction and probabilities.
    """

    results = predict_batch(network_traffic)

    if len(results) != 1:
        raise ValueError(
            "predict_attack accepts exactly one network connection. "
            "Use predict_batch for multiple connections."
        )

    return results[0]
