
from pathlib import Path
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)

from ..data.loader import load_raw_data
from ..data.preprocessing import prepare_dataset


# Target classes
CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def evaluate_model():

    # 1. Locate saved model
    project_root = Path(__file__).resolve().parents[2]

    model_path = (
        project_root
        / "models"
        / "artifacts"
        / "random_forest_pipeline.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    # 2. Load saved model
    print("Loading trained model...")

    model_pipeline = joblib.load(model_path)

    print("Model loaded successfully.")

    # 3. Load official test dataset
    _, test_df = load_raw_data()

    print("\nOfficial test dataset loaded.")
    print("Test dataset shape:", test_df.shape)

    # 4. Prepare test features and target
    X_test, y_test = prepare_dataset(test_df)

    print("\nTest features shape:", X_test.shape)
    print("Test target shape:", y_test.shape)

    # Display actual class distribution
    print("\nTest class distribution:")
    print(y_test.value_counts())

    # 5. Predict test data
    print("\nPredicting test data...")

    y_pred = model_pipeline.predict(X_test)

    print("Prediction completed!")

    # 6. Evaluate model
    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        labels=CLASS_NAMES,
        average="macro",
        zero_division=0
    )

    print("\n========== FINAL TEST RESULTS ==========")

    print("\nAccuracy:", round(accuracy, 4))
    print("Macro F1 Score:", round(macro_f1, 4))

    # 7. Classification report
    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=CLASS_NAMES,
            zero_division=0
        )
    )

    # 8. Confusion matrix
    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred,
            labels=CLASS_NAMES
        )
    )

    print("\nOfficial test evaluation completed successfully.")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "y_test": y_test,
        "y_pred": y_pred
    }


if __name__ == "__main__":
    results = evaluate_model()
