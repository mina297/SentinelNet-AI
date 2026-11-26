
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from ..data.loader import load_raw_data
from ..data.preprocessing import prepare_dataset


CLASS_NAMES = ["DoS", "Normal", "Probe", "R2L", "U2R"]


def evaluate_model(model, X_test, y_test, model_name):

    print("\n" + "=" * 70)
    print(f"{model_name} - OFFICIAL TEST EVALUATION")
    print("=" * 70)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=CLASS_NAMES,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        labels=CLASS_NAMES,
        average="weighted",
        zero_division=0
    )

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nClassification Report:")

    report = classification_report(
        y_test,
        predictions,
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print(report)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=CLASS_NAMES
    )

    print("Confusion Matrix:")
    print(
        pd.DataFrame(
            cm,
            index=CLASS_NAMES,
            columns=CLASS_NAMES
        )
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "predictions": predictions,
        "report": classification_report(
            y_test,
            predictions,
            labels=CLASS_NAMES,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": cm
    }


def main():

    print("=" * 70)
    print("SENTINELNET - FINAL MODEL EVALUATION")
    print("=" * 70)

    # Project paths
    project_root = Path(__file__).resolve().parents[2]

    artifacts_dir = project_root / "models" / "artifacts"
    reports_dir = project_root / "models" / "reports"

    reports_dir.mkdir(parents=True, exist_ok=True)

    xgb_path = artifacts_dir / "xgboost_pipeline.joblib"
    encoder_path = artifacts_dir / "xgboost_label_encoder.joblib"
    rf_path = artifacts_dir / "random_forest_pipeline.joblib"

    # Check XGBoost artifacts
    if not xgb_path.exists():
        raise FileNotFoundError(
            f"XGBoost model not found: {xgb_path}"
        )

    if not encoder_path.exists():
        raise FileNotFoundError(
            f"Label encoder not found: {encoder_path}"
        )

    # Load XGBoost model and label encoder
    print("\nLoading XGBoost model...")

    xgb_model = joblib.load(xgb_path)
    label_encoder = joblib.load(encoder_path)

    print("XGBoost model loaded successfully.")

    # Load official test dataset
    print("\nLoading official test dataset...")

    _, test_df = load_raw_data()

    print("Test dataset shape:", test_df.shape)

    # Prepare test features and target
    X_test, y_test = prepare_dataset(test_df)

    print("\nTest features shape:", X_test.shape)
    print("Test target shape:", y_test.shape)

    print("\nTest class distribution:")
    print(y_test.value_counts())

    # Check class compatibility
    if set(y_test.unique()) != set(label_encoder.classes_):
        raise ValueError(
            "Test classes do not match the classes used by LabelEncoder."
        )

    # XGBoost predictions are encoded integers
    xgb_encoded_predictions = xgb_model.predict(X_test)

    xgb_predictions = label_encoder.inverse_transform(
        xgb_encoded_predictions.astype(int)
    )

    # Evaluate XGBoost
    xgb_results = evaluate_model(
        model=type("PredictionWrapper", (), {
            "predict": lambda self, X: xgb_predictions
        })(),
        X_test=X_test,
        y_test=y_test,
        model_name="XGBoost Depth 4"
    )

    # Evaluate Random Forest if its artifact exists
    results = [xgb_results]

    if rf_path.exists():

        print("\nLoading Random Forest reference model...")

        rf_model = joblib.load(rf_path)

        print("Random Forest model loaded successfully.")

        rf_results = evaluate_model(
            model=rf_model,
            X_test=X_test,
            y_test=y_test,
            model_name="Random Forest"
        )

        results.append(rf_results)

    else:
        print("\nWARNING: Random Forest artifact was not found.")
        print("XGBoost will be evaluated without RF comparison.")

    # Final comparison table
    comparison_df = pd.DataFrame([
        {
            "model": result["model"],
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "weighted_f1": result["weighted_f1"]
        }
        for result in results
    ])

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison_df.sort_values(
            by="macro_f1",
            ascending=False
        ).to_string(index=False)
    )

    # Save comparison
    comparison_path = reports_dir / "final_model_comparison.csv"

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    # Save detailed classification reports
    for result in results:

        report_df = pd.DataFrame(
            result["report"]
        ).transpose()

        safe_name = result["model"].lower().replace(" ", "_")

        report_path = reports_dir / f"{safe_name}_test_report.csv"

        report_df.to_csv(report_path)

        cm_df = pd.DataFrame(
            result["confusion_matrix"],
            index=CLASS_NAMES,
            columns=CLASS_NAMES
        )

        cm_path = reports_dir / f"{safe_name}_confusion_matrix.csv"

        cm_df.to_csv(cm_path)

    print("\nReports saved successfully:")
    print(comparison_path)

    print("\nFinal evaluation completed successfully!")


if __name__ == "__main__":
    main()
