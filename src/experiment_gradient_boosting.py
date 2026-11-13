"""Standalone validation-grid experiment for Gradient Boosting rank classification.

This experiment leaves the production training pipeline and existing run folders
untouched. It chooses a configuration using only an inner validation split from
the existing training customers, then evaluates that selected configuration on
the fixed test set once.

Example:
    python -m src.experiment_gradient_boosting \
        --baseline-run outputs/runs/rank-gradient_boosting-20261105-195104-724058 \
        --output-dir outputs/experiments/gradient-boosting-grid-01
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from src.data_loader import load_data
from src.evaluate import COMMON_METRICS, evaluate_classification
from src.feature_engineering import FEATURE_COLUMNS, RANK_ORDER, build_customer_dataset, split_customer_dataset
from src.predict import build_model_input
from src.preprocessing import clean_transactions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRID = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3, 4],
    "min_samples_leaf": [5, 10, 20],
}
BASELINE_PARAMETERS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "random_state": 42,
}


def _evaluation_set_id(customer_ids):
    normalized = sorted(str(value) for value in customer_ids)
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()[:16]


def _json_default(value):
    """Convert NumPy scalar and array values when writing experiment metadata."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _evaluate(y_true, y_pred):
    metrics = evaluate_classification(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(RANK_ORDER), zero_division=0
    )
    result = {metric: float(metrics[metric]) for metric in COMMON_METRICS}
    for rank, p, r, score, count in zip(RANK_ORDER, precision, recall, f1, support):
        result[f"{rank}_Precision"] = float(p)
        result[f"{rank}_Recall"] = float(r)
        result[f"{rank}_F1"] = float(score)
        result[f"{rank}_Support"] = int(count)
    result["Confusion_Matrix"] = confusion_matrix(y_true, y_pred, labels=list(RANK_ORDER)).tolist()
    result["Missing_True_Classes"] = metrics["missing_true_classes"]
    result["Missing_Predicted_Classes"] = metrics["missing_predicted_classes"]
    return result


def _candidate_row(name, parameters, weighted, scores, fit_seconds):
    row = {
        "Candidate": name,
        "Sample_Weight": "balanced" if weighted else "none",
        **parameters,
        "fit_seconds": float(fit_seconds),
    }
    for key, value in scores.items():
        if key == "Confusion_Matrix":
            row[key] = json.dumps(value)
        elif isinstance(value, list):
            row[key] = json.dumps(value)
        else:
            row[key] = value
    return row


def run_experiment(
    transactions: pd.DataFrame,
    *,
    baseline_run: Path | str,
    output_dir: Path | str,
    test_size: float = 0.2,
    random_state: int = 42,
    validation_size: float = 0.2,
    validation_random_state: int = 2026,
):
    """Tune on a stratified slice of outer train, then evaluate selected once."""
    if not isinstance(transactions, pd.DataFrame) or transactions.empty:
        raise ValueError("transactions must be a non-empty DataFrame.")
    if not 0 < float(validation_size) < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    baseline_path = Path(baseline_run)
    if not baseline_path.is_absolute():
        baseline_path = PROJECT_ROOT / baseline_path
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    if output_path.exists():
        raise FileExistsError(f"Refusing to write into an existing experiment directory: {output_path}")
    experiments_root = (PROJECT_ROOT / "outputs" / "experiments").resolve()
    try:
        output_path.resolve().relative_to(experiments_root)
    except ValueError as exc:
        raise ValueError(f"Experiment output must be inside {experiments_root}") from exc
    if not baseline_path.is_dir():
        raise FileNotFoundError(f"Baseline run directory not found: {baseline_path}")

    clean = clean_transactions(transactions)
    dataset = build_customer_dataset(
        clean,
        prediction_days=90,
        test_size=test_size,
        random_state=random_state,
    )
    outer_train, test, thresholds = split_customer_dataset(
        dataset, test_size=test_size, random_state=random_state
    )
    if outer_train.empty or test.empty:
        raise ValueError("The fixed train/test split must contain both train and test customers.")

    fit_positions, validation_positions = train_test_split(
        np.arange(len(outer_train)),
        test_size=validation_size,
        random_state=validation_random_state,
        shuffle=True,
        stratify=outer_train["CustomerRank"],
    )
    fit_customers = outer_train.iloc[fit_positions]
    validation_customers = outer_train.iloc[validation_positions]
    X_fit = build_model_input(fit_customers)
    X_validation = build_model_input(validation_customers)
    y_fit = fit_customers["CustomerRank"].to_numpy()
    y_validation = validation_customers["CustomerRank"].to_numpy()

    rows = []
    validation_predictions = {}

    # Baseline is measured on the exact same inner validation set.
    baseline_model = GradientBoostingClassifier(**BASELINE_PARAMETERS)
    started = time.perf_counter()
    baseline_model.fit(X_fit, y_fit)
    fit_seconds = time.perf_counter() - started
    baseline_prediction = baseline_model.predict(X_validation)
    baseline_scores = _evaluate(y_validation, baseline_prediction)
    rows.append(_candidate_row("baseline", BASELINE_PARAMETERS, False, baseline_scores, fit_seconds))
    validation_predictions["baseline"] = baseline_prediction
    print(
        f"Validation baseline: Macro F1={baseline_scores['F1_Macro']:.4f}, "
        f"Balanced Accuracy={baseline_scores['Balanced_Accuracy']:.4f}",
        flush=True,
    )

    parameter_grid = list(ParameterGrid(DEFAULT_GRID))
    for index, grid_parameters in enumerate(parameter_grid, start=1):
        parameters = {
            **grid_parameters,
            "min_samples_split": BASELINE_PARAMETERS["min_samples_split"],
            "random_state": random_state,
        }
        candidate_id = f"balanced-{index:02d}"
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_fit)
        model = GradientBoostingClassifier(**parameters)
        started = time.perf_counter()
        model.fit(X_fit, y_fit, sample_weight=sample_weights)
        fit_seconds = time.perf_counter() - started
        prediction = model.predict(X_validation)
        scores = _evaluate(y_validation, prediction)
        rows.append(_candidate_row(candidate_id, parameters, True, scores, fit_seconds))
        validation_predictions[candidate_id] = prediction
        print(
            f"Validation {index:02d}/{len(parameter_grid)} {parameters}: "
            f"Macro F1={scores['F1_Macro']:.4f}, Balanced Accuracy={scores['Balanced_Accuracy']:.4f}",
            flush=True,
        )

    validation_results = pd.DataFrame(rows)
    validation_results["_min_class_recall"] = validation_results[
        [f"{rank}_Recall" for rank in RANK_ORDER]
    ].min(axis=1)
    validation_results = validation_results.sort_values(
        ["F1_Macro", "Balanced_Accuracy", "_min_class_recall", "Accuracy"],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    validation_results["Validation_Rank"] = np.arange(1, len(validation_results) + 1)
    validation_results = validation_results.drop(columns="_min_class_recall")
    best_row = validation_results.iloc[0]
    best_id = str(best_row["Candidate"])
    best_weighted = str(best_row["Sample_Weight"]) == "balanced"
    best_parameters = {
        key: (int(best_row[key]) if key in {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "random_state"} else float(best_row[key]))
        for key in (
            "n_estimators", "learning_rate", "max_depth", "min_samples_split",
            "min_samples_leaf", "random_state",
        )
    }

    # No test labels, predictions, or metrics are read until the validation winner is fixed.
    final_model = GradientBoostingClassifier(**best_parameters)
    final_weights = compute_sample_weight(class_weight="balanced", y=outer_train["CustomerRank"]) if best_weighted else None
    X_outer_train = build_model_input(outer_train)
    X_test = build_model_input(test)
    final_model.fit(X_outer_train, outer_train["CustomerRank"], sample_weight=final_weights)
    selected_test_predictions = final_model.predict(X_test)
    selected_test_scores = _evaluate(test["CustomerRank"], selected_test_predictions)

    with (baseline_path / "run_metadata.json").open("r", encoding="utf-8") as stream:
        baseline_metadata = json.load(stream)
    if not isinstance(baseline_metadata, dict) or baseline_metadata.get("model") != "Gradient Boosting":
        raise ValueError("The supplied baseline run metadata is not a Gradient Boosting run.")
    test_id = _evaluation_set_id(test["CustomerID"])
    if test_id != baseline_metadata.get("evaluation_set_id"):
        raise ValueError("Recreated test split does not match the saved baseline evaluation_set_id.")
    baseline_predictions_path = baseline_path / "test_predictions.csv"
    baseline_metrics_path = baseline_path / "ranking_metrics.csv"
    baseline_predictions = pd.read_csv(baseline_predictions_path)
    baseline_metrics = pd.read_csv(baseline_metrics_path)
    if len(baseline_predictions) != len(test) or len(baseline_metrics) != 1:
        raise ValueError("Baseline test artifacts are incomplete or have unexpected sizes.")
    if (
        "Model" not in baseline_metrics.columns
        or baseline_metrics.iloc[0]["Model"] != "Gradient Boosting"
        or "Evaluation_Set_ID" not in baseline_metrics.columns
        or str(baseline_metrics.iloc[0]["Evaluation_Set_ID"]) != test_id
    ):
        raise ValueError("Baseline metrics do not identify the expected Gradient Boosting test set.")
    recreated_test_ids = test["CustomerID"].map(lambda value: str(int(float(value)))).tolist()
    saved_test_ids = baseline_predictions["CustomerID"].map(lambda value: str(int(float(value)))).tolist()
    if saved_test_ids != recreated_test_ids:
        raise ValueError("Saved baseline predictions do not match the recreated test customer order.")
    if baseline_predictions["CustomerRank_true"].astype(str).tolist() != test["CustomerRank"].astype(str).tolist():
        raise ValueError("Saved baseline true labels do not match the recreated test set.")
    baseline_test_scores = _evaluate(
        test["CustomerRank"], baseline_predictions["CustomerRank_predicted"].astype(str)
    )
    comparison = []
    for name, scores in (("Baseline", baseline_test_scores), ("Selected", selected_test_scores)):
        comparison.append({
            "Model": name,
            "Candidate": "baseline" if name == "Baseline" else best_id,
            "Sample_Weight": "none" if name == "Baseline" else ("balanced" if best_weighted else "none"),
            **{key: value for key, value in scores.items() if key != "Confusion_Matrix" and not isinstance(value, list)},
            "Confusion_Matrix": json.dumps(scores["Confusion_Matrix"]),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.mkdir()
    validation_results.to_csv(output_path / "validation_results.csv", index=False, encoding="utf-8", mode="x")
    pd.DataFrame(comparison).to_csv(output_path / "test_comparison.csv", index=False, encoding="utf-8", mode="x")
    pd.DataFrame(
        {
            "CustomerID": test["CustomerID"].to_numpy(),
            "CustomerRank_true": test["CustomerRank"].to_numpy(),
            "Baseline_predicted": baseline_predictions["CustomerRank_predicted"].astype(str).to_numpy(),
            "Selected_predicted": selected_test_predictions,
        }
    ).to_csv(output_path / "test_predictions_comparison.csv", index=False, encoding="utf-8", mode="x")
    with (output_path / "test_confusion_matrices.json").open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "labels": list(RANK_ORDER),
                "Baseline": baseline_test_scores["Confusion_Matrix"],
                "Selected": selected_test_scores["Confusion_Matrix"],
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )

    validation_customer_ids = validation_customers["CustomerID"].tolist()
    metadata = {
        "experiment": "Gradient Boosting sample-weight and parameter grid",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_run": str(baseline_path.resolve().relative_to(PROJECT_ROOT)),
        "baseline_evaluation_set_id": baseline_metadata["evaluation_set_id"],
        "outer_test_size": test_size,
        "outer_random_state": random_state,
        "validation_size_from_outer_train": validation_size,
        "validation_random_state": validation_random_state,
        "validation_strategy": "stratified random split of outer training customers only",
        "n_outer_train_customers": len(outer_train),
        "n_validation_customers": len(validation_customers),
        "n_inner_fit_customers": len(fit_customers),
        "n_test_customers": len(test),
        "validation_evaluation_set_id": _evaluation_set_id(validation_customer_ids),
        "rank_thresholds_from_existing_outer_train": thresholds.as_dict(),
        "feature_columns": list(FEATURE_COLUMNS),
        "sklearn_version": sklearn.__version__,
        "selection_order": ["F1_Macro", "Balanced_Accuracy", "minimum class recall", "Accuracy"],
        "selected_candidate": best_id,
        "selected_sample_weight": "balanced" if best_weighted else "none",
        "selected_parameters": best_parameters,
        "selected_validation_metrics": {
            column: (json.loads(best_row[column]) if column == "Confusion_Matrix" else best_row[column])
            for column in validation_results.columns
            if column in COMMON_METRICS
            or column.endswith(("_Precision", "_Recall", "_F1", "_Support"))
            or column == "Confusion_Matrix"
        },
        "test_evaluated_after_validation_selection": True,
        "production_pipeline_changed": False,
        "test_evaluation_count_for_selected_candidate": 1,
    }
    with (output_path / "experiment_metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, default=_json_default)
    print(f"Selected validation candidate: {best_id}; weights={metadata['selected_sample_weight']}; params={best_parameters}")
    print(pd.DataFrame(comparison).to_string(index=False))
    print(f"Experiment artifacts saved to {output_path}")
    return {
        "validation_results": validation_results,
        "test_comparison": pd.DataFrame(comparison),
        "metadata": metadata,
        "output_dir": output_path,
        "validation_predictions": validation_predictions,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run an isolated Gradient Boosting validation grid.")
    parser.add_argument("--baseline-run", required=True, help="Existing Gradient Boosting run folder used for comparison.")
    parser.add_argument("--output-dir", required=True, help="New folder under outputs/experiments; must not exist.")
    parser.add_argument("--data", default=None, help="Path to Online Retail.xlsx; defaults to the project raw file.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--validation-random-state", type=int, default=2026)
    args = parser.parse_args(argv)
    result = run_experiment(
        load_data(args.data),
        baseline_run=args.baseline_run,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        validation_size=args.validation_size,
        validation_random_state=args.validation_random_state,
    )
    return result


if __name__ == "__main__":
    main()
