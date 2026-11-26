
from pathlib import Path

import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
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
N_SPLITS = 5


EXPERIMENTS = [
    {
        "name": "XGBoost Baseline",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1
    },
    {
        "name": "XGBoost Depth 4",
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.1
    }
]


def create_model(config, num_classes):

    classifier = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
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

    model = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", classifier)
    ])

    return model


def run_cross_validation():

    print("=" * 70)
    print("XGBOOST 5-FOLD CROSS VALIDATION")
    print("=" * 70)

    # Load training data only
    train_df, _ = load_raw_data()

    print("\nTraining dataset loaded.")
    print("Dataset shape:", train_df.shape)

    X, y = prepare_dataset(train_df)

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    class_names = list(label_encoder.classes_)

    print("\nClass mapping:")

    for index, class_name in enumerate(class_names):
        print(f"{index} = {class_name}")

    # Stratified K-Fold
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    all_results = []
    all_class_results = []

    for config in EXPERIMENTS:

        print("\n" + "=" * 70)
        print("Experiment:", config["name"])
        print("=" * 70)

        model = create_model(
            config,
            num_classes=len(class_names)
        )

        fold_accuracies = []
        fold_macro_f1 = []

        # Store metrics for each class across folds
        class_metrics = {
            class_name: {
                "precision": [],
                "recall": [],
                "f1-score": []
            }
            for class_name in class_names
        }

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X, y_encoded), start=1
        ):

            print(f"\nTraining Fold {fold}/{N_SPLITS}...")

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_train = y_encoded[train_idx]
            y_val = y_encoded[val_idx]

            # Calculate balanced weights from training fold only
            sample_weights = compute_sample_weight(
                class_weight="balanced",
                y=y_train
            )

            # Create a fresh model for this fold
            fold_model = clone(model)

            fold_model.fit(
                X_train,
                y_train,
                classifier__sample_weight=sample_weights
            )

            predictions = fold_model.predict(X_val)

            # Overall metrics
            accuracy = accuracy_score(
                y_val,
                predictions
            )

            macro_f1 = f1_score(
                y_val,
                predictions,
                average="macro"
            )

            fold_accuracies.append(accuracy)
            fold_macro_f1.append(macro_f1)

            print(f"Fold {fold} Accuracy: {accuracy:.4f}")
            print(f"Fold {fold} Macro F1: {macro_f1:.4f}")

            # Class-wise metrics
            report = classification_report(
                y_val,
                predictions,
                labels=list(range(len(class_names))),
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )

            print("\nClass-wise Metrics:")

            for class_name in class_names:

                precision = report[class_name]["precision"]
                recall = report[class_name]["recall"]
                class_f1 = report[class_name]["f1-score"]

                class_metrics[class_name]["precision"].append(
                    precision
                )

                class_metrics[class_name]["recall"].append(
                    recall
                )

                class_metrics[class_name]["f1-score"].append(
                    class_f1
                )

                all_class_results.append({
                    "experiment": config["name"],
                    "fold": fold,
                    "class": class_name,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": class_f1,
                    "support": report[class_name]["support"]
                })

                print(
                    f"{class_name:10} | "
                    f"Precision: {precision:.4f} | "
                    f"Recall: {recall:.4f} | "
                    f"F1: {class_f1:.4f}"
                )

        # Overall experiment summary
        mean_accuracy = sum(fold_accuracies) / N_SPLITS
        mean_macro_f1 = sum(fold_macro_f1) / N_SPLITS

        std_accuracy = pd.Series(fold_accuracies).std(ddof=0)
        std_macro_f1 = pd.Series(fold_macro_f1).std(ddof=0)

        print("\n" + "-" * 70)
        print("Experiment Summary:", config["name"])
        print("-" * 70)

        print(f"Mean Accuracy: {mean_accuracy:.4f}")
        print(f"Std Accuracy: {std_accuracy:.4f}")
        print(f"Mean Macro F1: {mean_macro_f1:.4f}")
        print(f"Std Macro F1: {std_macro_f1:.4f}")

        all_results.append({
            "experiment": config["name"],
            "n_estimators": config["n_estimators"],
            "max_depth": config["max_depth"],
            "learning_rate": config["learning_rate"],
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "mean_macro_f1": mean_macro_f1,
            "std_macro_f1": std_macro_f1
        })

        # Average class-wise metrics
        print("\nAverage Class-wise Metrics:")

        for class_name in class_names:

            mean_precision = sum(
                class_metrics[class_name]["precision"]
            ) / N_SPLITS

            mean_recall = sum(
                class_metrics[class_name]["recall"]
            ) / N_SPLITS

            mean_f1 = sum(
                class_metrics[class_name]["f1-score"]
            ) / N_SPLITS

            print(
                f"{class_name:10} | "
                f"Precision: {mean_precision:.4f} | "
                f"Recall: {mean_recall:.4f} | "
                f"F1: {mean_f1:.4f}"
            )

    # Final comparison
    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by="mean_macro_f1",
        ascending=False
    )

    print("\n" + "=" * 70)
    print("FINAL CROSS VALIDATION COMPARISON")
    print("=" * 70)

    print(results_df.to_string(index=False))

    best_experiment = results_df.iloc[0]

    print("\nBest Experiment:", best_experiment["experiment"])

    print(
        "Best Mean Macro F1:",
        f"{best_experiment['mean_macro_f1']:.4f}"
    )

    # Final class-wise comparison
    class_results_df = pd.DataFrame(all_class_results)

    class_summary_df = (
        class_results_df
        .groupby(["experiment", "class"])
        .agg(
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1_score", "mean"),
            std_f1=("f1_score", "std")
        )
        .reset_index()
    )

    print("\n" + "=" * 70)
    print("FINAL CLASS-WISE CROSS VALIDATION SUMMARY")
    print("=" * 70)

    print(class_summary_df.to_string(index=False))

    # Save reports
    project_root = Path(__file__).resolve().parents[2]

    reports_dir = project_root / "models" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = reports_dir / "xgboost_cv_results.csv"
    class_details_path = reports_dir / "xgboost_cv_class_details.csv"
    class_summary_path = reports_dir / "xgboost_cv_class_summary.csv"

    results_df.to_csv(comparison_path, index=False)
    class_results_df.to_csv(class_details_path, index=False)
    class_summary_df.to_csv(class_summary_path, index=False)

    print("\nReports saved at:")
    print(comparison_path)
    print(class_details_path)
    print(class_summary_path)

    print("\nCross-validation completed successfully!")

    return results_df, class_summary_df


if __name__ == "__main__":
    run_cross_validation()
