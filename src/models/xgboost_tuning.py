
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
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


EXPERIMENTS = [
    {
        "name": "XGBoost Baseline",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1
    },
    {
        "name": "Depth 4",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.1
    },
    {
        "name": "Depth 8",
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.1
    },
    {
        "name": "Lower Learning Rate",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05
    },
    {
        "name": "Higher Learning Rate",
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.2
    },
    {
        "name": "More Estimators",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.1
    },
    {
        "name": "Fewer Estimators",
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1
    }
]


def run_tuning():

    # 1. Load training data only
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

    # 4. Encode target labels
    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    print("\nClass mapping:")

    for index, class_name in enumerate(label_encoder.classes_):
        print(index, "=", class_name)

    # 5. Calculate balanced sample weights
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train_encoded
    )

    results = []

    best_f1 = -1
    best_name = None
    best_model = None
    best_predictions = None

    # 6. Random Forest reference baseline
    print("\n" + "=" * 60)
    print("Random Forest Reference Baseline")
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

    rf_macro_f1 = f1_score(
        y_val,
        rf_predictions,
        labels=CLASS_NAMES,
        average="macro",
        zero_division=0
    )

    rf_accuracy = accuracy_score(y_val, rf_predictions)

    results.append({
        "experiment": "Random Forest Baseline",
        "n_estimators": 200,
        "max_depth": None,
        "learning_rate": None,
        "accuracy": rf_accuracy,
        "macro_f1": rf_macro_f1
    })

    print("Random Forest Macro F1:", round(rf_macro_f1, 4))

    # 7. Run XGBoost experiments
    for config in EXPERIMENTS:

        print("\n" + "=" * 60)
        print("Experiment:", config["name"])
        print("=" * 60)

        xgb_pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                (
                    "classifier",
                    XGBClassifier(
                        objective="multi:softprob",
                        num_class=len(label_encoder.classes_),
                        n_estimators=config["n_estimators"],
                        max_depth=config["max_depth"],
                        learning_rate=config["learning_rate"],
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

        print("Training XGBoost...")

        xgb_pipeline.fit(
            X_train,
            y_train_encoded,
            classifier__sample_weight=sample_weights
        )

        print("Training completed.")

        predictions_encoded = xgb_pipeline.predict(X_val)

        predictions = label_encoder.inverse_transform(
            predictions_encoded.astype(int)
        )

        accuracy = accuracy_score(y_val, predictions)

        macro_f1 = f1_score(
            y_val,
            predictions,
            labels=CLASS_NAMES,
            average="macro",
            zero_division=0
        )

        print("\nAccuracy:", round(accuracy, 4))
        print("Macro F1:", round(macro_f1, 4))

        results.append({
            "experiment": config["name"],
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"],
            "learning_rate": config["learning_rate"],
            "accuracy": accuracy,
            "macro_f1": macro_f1
        })

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_name = config["name"]
            best_model = xgb_pipeline
            best_predictions = predictions

    # 8. Compare all experiments
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="macro_f1",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("FINAL XGBOOST TUNING COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "macro_f1": "{:.4f}".format
            }
        )
    )

    # 9. Best XGBoost classification report
    print("\nBest XGBoost Experiment:", best_name)
    print("Best Macro F1:", round(best_f1, 4))

    print("\nBest XGBoost Classification Report:")

    print(
        classification_report(
            y_val,
            best_predictions,
            labels=CLASS_NAMES,
            zero_division=0
        )
    )

    # 10. Save experiment report
    project_root = Path(__file__).resolve().parents[2]

    report_dir = project_root / "models" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "xgboost_tuning.csv"

    results_df.to_csv(report_path, index=False)

    print("\nReport saved at:")
    print(report_path)

    print("\nXGBoost tuning completed successfully!")

    return results_df, best_model


if __name__ == "__main__":

    results, best_model = run_tuning()
