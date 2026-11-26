
from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)

from ..data.loader import load_raw_data
from ..data.preprocessing import prepare_dataset
from ..features.preprocessor import create_preprocessor


RANDOM_STATE = 42

CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def train_model():

    # 1. Load the original training dataset
    train_df, _ = load_raw_data()

    print("Dataset loaded successfully.")
    print("Original training shape:", train_df.shape)

    # 2. Prepare features and mapped target
    X, y = prepare_dataset(train_df)

    # Keep original attack names for detailed analysis
    raw_attack_types = (
        train_df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.rstrip(".")
    )

    print("\nFeatures shape:", X.shape)
    print("Target shape:", y.shape)

    print("\nClass distribution:")
    print(y.value_counts())

    # 3. Split data into training and validation sets
    (
        X_train,
        X_val,
        y_train,
        y_val,
        raw_train,
        raw_val
    ) = train_test_split(
        X,
        y,
        raw_attack_types,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTrain size:", X_train.shape[0])
    print("Validation size:", X_val.shape[0])

    # 4. Build the complete ML pipeline
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ]
    )

    # 5. Train the model
    print("\nTraining Random Forest...")

    model_pipeline.fit(X_train, y_train)

    print("Training completed!")

    # 6. Save the trained pipeline
    project_root = Path(__file__).resolve().parents[2]

    model_dir = project_root / "models" / "artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "random_forest_pipeline.joblib"

    joblib.dump(model_pipeline, model_path)

    print(f"\nModel saved successfully at: {model_path}")

    print("\nModel classes:")
    print(model_pipeline.named_steps["classifier"].classes_)

    # 7. Evaluate on validation data
    print("\nPredicting validation data...")

    y_pred = model_pipeline.predict(X_val)

    print("Validation predictions completed!")

    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro"
    )

    print("\nMacro F1 Score:", round(macro_f1, 4))

    print("\nClassification Report:")

    print(
        classification_report(
            y_val,
            y_pred,
            labels=CLASS_NAMES,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_val,
            y_pred,
            labels=CLASS_NAMES
        )
    )

    # 8. Analyze performance by original attack type
    attack_analysis = pd.DataFrame({
        "actual_attack": raw_val.values,
        "actual_class": y_val.values,
        "predicted_class": y_pred
    })

    attack_analysis["correct"] = (
        attack_analysis["actual_class"]
        == attack_analysis["predicted_class"]
    )

    attack_summary = (
        attack_analysis
        .groupby("actual_attack")
        .agg(
            samples=("correct", "count"),
            correct=("correct", "sum"),
            recall=("correct", "mean")
        )
        .sort_values("recall")
    )

    print("\n========== PERFORMANCE BY ORIGINAL ATTACK TYPE ==========")

    print(attack_summary.to_string())

    print("\nTraining and validation analysis completed successfully.")

    return model_pipeline


if __name__ == "__main__":

    trained_model = train_model()
