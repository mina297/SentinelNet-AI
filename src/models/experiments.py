
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

from ..data.loader import load_raw_data
from ..data.preprocessing import prepare_dataset
from ..features.preprocessor import create_preprocessor


RANDOM_STATE = 42

CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]


EXPERIMENTS = [
    {
        "name": "Random Forest Baseline",
        "classifier": "random_forest",
        "n_estimators": 200,
        "class_weight": "balanced"
    },
    {
        "name": "Extra Trees Balanced",
        "classifier": "extra_trees",
        "n_estimators": 200,
        "class_weight": "balanced"
    },
    {
        "name": "Extra Trees More Trees",
        "classifier": "extra_trees",
        "n_estimators": 300,
        "class_weight": "balanced"
    },
    {
        "name": "Extra Trees No Class Weight",
        "classifier": "extra_trees",
        "n_estimators": 200,
        "class_weight": None
    }
]


def create_classifier(config):

    if config["classifier"] == "random_forest":

        return RandomForestClassifier(
            n_estimators=config["n_estimators"],
            class_weight=config["class_weight"],
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    if config["classifier"] == "extra_trees":

        return ExtraTreesClassifier(
            n_estimators=config["n_estimators"],
            class_weight=config["class_weight"],
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

    raise ValueError("Unknown classifier type")


def run_experiments():

    # 1. Load training data
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

    best_f1 = -1
    best_name = None
    best_model = None

    # 4. Run experiments
    for config in EXPERIMENTS:

        print("\n" + "=" * 55)
        print("Experiment:", config["name"])
        print("=" * 55)

        model = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("classifier", create_classifier(config))
            ]
        )

        print("Training model...")
        model.fit(X_train, y_train)

        print("Predicting validation data...")
        y_pred = model.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)

        macro_f1 = f1_score(
            y_val,
            y_pred,
            labels=CLASS_NAMES,
            average="macro",
            zero_division=0
        )

        weighted_f1 = f1_score(
            y_val,
            y_pred,
            average="weighted",
            zero_division=0
        )

        results.append({
            "experiment": config["name"],
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1
        })

        print("\nAccuracy:", round(accuracy, 4))
        print("Macro F1:", round(macro_f1, 4))
        print("Weighted F1:", round(weighted_f1, 4))

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_name = config["name"]
            best_model = model

    # 5. Compare results
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

    # 6. Best model report
    print("\nBest Experiment:", best_name)
    print("Best Macro F1:", round(best_f1, 4))

    best_predictions = best_model.predict(X_val)

    print("\nBest Model Classification Report:")

    print(
        classification_report(
            y_val,
            best_predictions,
            labels=CLASS_NAMES,
            zero_division=0
        )
    )

    # 7. Save comparison report
    project_root = Path(__file__).resolve().parents[2]

    report_dir = project_root / "models" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "tree_model_comparison.csv"

    results_df.to_csv(report_path, index=False)

    print("\nComparison report saved at:")
    print(report_path)

    print("\nAll experiments completed successfully!")

    return results_df, best_model


if __name__ == "__main__":

    results, best_model = run_experiments()
