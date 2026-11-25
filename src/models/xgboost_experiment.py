
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

from xgboost import XGBClassifier

from ..data.loader import load_raw_data
from ..data.preprocessing import prepare_dataset
from ..features.preprocessor import create_preprocessor


RANDOM_STATE = 42

CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def evaluate_predictions(name, y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\nModel:", name)
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=CLASS_NAMES,
            zero_division=0
        )
    )

    return {
        "model": name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


def run_xgboost_experiment():

    # 1. Load training dataset
    train_df, _ = load_raw_data()

    print("Training dataset loaded.")
    print("Dataset shape:", train_df.shape)

    # 2. Prepare features and target
    X, y = prepare_dataset(train_df)

    # 3. Fixed validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Validation samples:", len(X_val))

    results = []

    # 4. Random Forest baseline
    print("\n" + "=" * 60)
    print("Training Random Forest Baseline")
    print("=" * 60)

    rf_pipeline = Pipeline(
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

    rf_pipeline.fit(X_train, y_train)

    rf_predictions = rf_pipeline.predict(X_val)

    rf_result = evaluate_predictions(
        "Random Forest Baseline",
        y_val,
        rf_predictions
    )

    results.append(rf_result)

    # 5. Encode target labels for XGBoost
    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    print("\nXGBoost class mapping:")

    for index, class_name in enumerate(label_encoder.classes_):
        print(index, "=", class_name)

    # 6. Calculate balanced sample weights
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train_encoded
    )

    # 7. Create XGBoost pipeline
    print("\n" + "=" * 60)
    print("Training XGBoost")
    print("=" * 60)

    xgb_pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=len(label_encoder.classes_),
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="mlogloss",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ]
    )

    xgb_pipeline.fit(
        X_train,
        y_train_encoded,
        classifier__sample_weight=sample_weights
    )

    print("XGBoost training completed!")

    # 8. Predict validation data
    print("\nPredicting validation data...")

    xgb_predictions_encoded = xgb_pipeline.predict(X_val)

    xgb_predictions = label_encoder.inverse_transform(
        xgb_predictions_encoded.astype(int)
    )

    print("XGBoost predictions completed!")

    xgb_result = evaluate_predictions(
        "XGBoost",
        y_val,
        xgb_predictions
    )

    results.append(xgb_result)

    # 9. Compare models
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="macro_f1",
        ascending=False
    )

    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON")
    print("=" * 65)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "macro_f1": "{:.4f}".format,
                "weighted_f1": "{:.4f}".format
            }
        )
    )

    # 10. Save comparison report
    project_root = Path(__file__).resolve().parents[2]

    report_dir = project_root / "models" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "xgboost_comparison.csv"

    results_df.to_csv(report_path, index=False)

    print("\nComparison report saved at:")
    print(report_path)

    print("\nExperiment completed successfully!")

    return results_df


if __name__ == "__main__":
    results = run_xgboost_experiment()
