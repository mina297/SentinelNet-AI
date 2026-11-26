
from pathlib import Path

import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

from ..data.loader import load_raw_data
from ..data.preprocessing import prepare_dataset
from ..features.preprocessor import create_preprocessor


RANDOM_STATE = 42


def create_xgboost_model(num_classes):

    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", classifier)
    ])

    return model


def train_xgboost():

    print("=" * 70)
    print("FINAL XGBOOST MODEL TRAINING")
    print("=" * 70)

    # Load official training dataset only
    train_df, _ = load_raw_data()

    print("\nTraining dataset loaded.")
    print("Dataset shape:", train_df.shape)

    # Prepare features and target
    X, y = prepare_dataset(train_df)

    print("\nFeatures shape:", X.shape)
    print("Target shape:", y.shape)

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\nClass mapping:")

    for index, class_name in enumerate(label_encoder.classes_):
        print(f"{index} = {class_name}")

    # Display class distribution
    print("\nClass distribution:")
    print(y.value_counts())

    # Calculate balanced sample weights
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_encoded
    )

    print("\nSample weights calculated.")

    # Create model
    model = create_xgboost_model(
        num_classes=len(label_encoder.classes_)
    )

    print("\nModel configuration:")
    print("Algorithm: XGBoost")
    print("Estimators: 300")
    print("Max Depth: 4")
    print("Learning Rate: 0.1")
    print("Subsample: 0.8")
    print("Column Sample: 0.8")

    # Train on the complete training dataset
    print("\nTraining XGBoost on full training dataset...")

    model.fit(
        X,
        y_encoded,
        classifier__sample_weight=sample_weights
    )

    print("Training completed successfully.")

    # Save model artifacts
    project_root = Path(__file__).resolve().parents[2]

    artifacts_dir = project_root / "models" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "xgboost_pipeline.joblib"
    encoder_path = artifacts_dir / "xgboost_label_encoder.joblib"

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)

    print("\nArtifacts saved:")
    print(model_path)
    print(encoder_path)

    print("\n" + "=" * 70)
    print("FINAL XGBOOST TRAINING SUMMARY")
    print("=" * 70)

    print("Training samples:", len(X))
    print("Number of features:", X.shape[1])
    print("Number of classes:", len(label_encoder.classes_))
    print("Model: XGBoost")
    print("Max Depth: 4")
    print("Estimators: 300")

    print("\nFinal XGBoost model is ready for evaluation.")

    return model, label_encoder


if __name__ == "__main__":
    train_xgboost()
