"""Train and evaluate a Random Forest customer rank classifier."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import pandas as pd

import src.data_loader as data_loader
from src.evaluate import evaluate_classification, make_model_result, write_model_results
from src.feature_engineering import FEATURE_COLUMNS, RANK_ORDER, build_customer_dataset, split_customer_dataset
from src.models.random_forest_model import RandomForestRankClassifier
from src.predict import build_model_input
from src.preprocessing import clean_transactions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_N_ESTIMATORS = 100
DEFAULT_MAX_DEPTH = 5
DEFAULT_MIN_SAMPLES_SPLIT = 20
DEFAULT_MIN_SAMPLES_LEAF = 10
DEFAULT_MAX_FEATURES = "sqrt"
DEFAULT_BOOTSTRAP = True
DEFAULT_CLASS_WEIGHT = None
DEFAULT_N_JOBS = -1


def train_random_forest_rank_model(
    dataset: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    min_samples_split: int = DEFAULT_MIN_SAMPLES_SPLIT,
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
    max_features: str = DEFAULT_MAX_FEATURES,
    bootstrap: bool = DEFAULT_BOOTSTRAP,
    class_weight=None,
    n_jobs: int = DEFAULT_N_JOBS,
):
    """Fit on the customer-level train split and evaluate its fixed holdout."""
    if not isinstance(dataset, pd.DataFrame) or dataset.empty:
        raise ValueError("dataset must be a non-empty customer DataFrame.")
    required = {"CustomerID", "FutureSpend", "CustomerRank", "ReturnedNext90Days"}
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise KeyError(f"Missing target or identifier columns: {', '.join(missing)}")
    if dataset["CustomerID"].isna().any() or dataset["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must be present and unique before model training.")
    if dataset["FutureSpend"].isna().any():
        raise ValueError("FutureSpend must be present for train-only ranking thresholds.")
    missing_features = [column for column in FEATURE_COLUMNS if column not in dataset.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns: {', '.join(missing_features)}")

    train, test, thresholds = split_customer_dataset(
        dataset, test_size=test_size, random_state=random_state
    )
    if train.empty or test.empty:
        raise ValueError("Random Forest training and evaluation both require customer rows.")

    X_train = build_model_input(train)
    X_test = build_model_input(test)
    model = RandomForestRankClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    started = time.perf_counter()
    model.fit(X_train, train["CustomerRank"])
    training_time = time.perf_counter() - started

    evaluation_set_id = _evaluation_set_id(test["CustomerID"])
    started = time.perf_counter()
    predictions = model.predict(X_test)
    prediction_time = time.perf_counter() - started
    metrics = evaluate_classification(test["CustomerRank"], predictions)
    result_row = make_model_result(
        "Random Forest",
        "Classification",
        metrics,
        training_time=training_time,
        prediction_time=prediction_time,
        evaluation_set_id=evaluation_set_id,
    )
    test_predictions = pd.DataFrame(
        {
            "CustomerID": test["CustomerID"].to_numpy(),
            "CustomerRank_true": test["CustomerRank"].to_numpy(),
            "CustomerRank_predicted": predictions,
        }
    )

    train_ids = train["CustomerID"].tolist()
    test_ids = test["CustomerID"].tolist()
    if set(train_ids) & set(test_ids):
        raise RuntimeError("Customer ID leakage detected between train and test.")
    metadata = {
        "model": "Random Forest",
        "model_type": "Classification",
        "random_state": random_state,
        "test_size": test_size,
        "n_train_customers": len(train),
        "n_test_customers": len(test),
        "evaluation_set_id": evaluation_set_id,
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_importances": [
            {"Feature": feature, "Importance": float(importance)}
            for feature, importance in zip(FEATURE_COLUMNS, model.feature_importances_)
        ],
        "rank_thresholds": thresholds.as_dict(),
        "model_parameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "bootstrap": bootstrap,
            "class_weight": class_weight,
            "random_state": random_state,
            "n_jobs": n_jobs,
        },
        "train_rank_distribution": _rank_distribution(train["CustomerRank"]),
        "test_rank_distribution": _rank_distribution(test["CustomerRank"]),
        "training_time_seconds": float(training_time),
        "prediction_time_seconds": float(prediction_time),
        "prediction_window_start": _attr_iso(dataset, "prediction_start"),
        "observation_end": _attr_iso(dataset, "observation_end"),
        "temporal_scope": "single cutoff; not a multi-period backtest",
    }
    return {
        "model": model,
        "metrics": metrics,
        "result_row": result_row,
        "predictions": test_predictions,
        "metadata": metadata,
        "train_customer_ids": train_ids,
        "test_customer_ids": test_ids,
        "thresholds": thresholds,
    }


def save_experiment(result: dict, output_dir: Path | str):
    """Write all artifacts into a new run directory; refuse every overwrite."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to write into an existing run directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    joblib.dump(result["model"], destination / "model.joblib")
    write_model_results([result["result_row"]], destination / "ranking_metrics.csv")
    result["predictions"].to_csv(
        destination / "test_predictions.csv", index=False, encoding="utf-8", mode="x"
    )
    with (destination / "run_metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(result["metadata"], stream, ensure_ascii=False, indent=2)
    return destination


def run_training(
    transactions: pd.DataFrame,
    *,
    prediction_days: int = 90,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    max_depth: int = DEFAULT_MAX_DEPTH,
    min_samples_split: int = DEFAULT_MIN_SAMPLES_SPLIT,
    min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
    max_features: str = DEFAULT_MAX_FEATURES,
    bootstrap: bool = DEFAULT_BOOTSTRAP,
    class_weight=None,
    n_jobs: int = DEFAULT_N_JOBS,
):
    """Clean transactions, build the shared dataset, and run the classifier."""
    clean = clean_transactions(transactions)
    dataset = build_customer_dataset(
        clean,
        prediction_days=prediction_days,
        test_size=test_size,
        random_state=random_state,
    )
    result = train_random_forest_rank_model(
        dataset,
        test_size=test_size,
        random_state=random_state,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        class_weight=class_weight,
        n_jobs=n_jobs,
    )
    result["dataset"] = dataset
    return result


def _evaluation_set_id(customer_ids):
    normalized = sorted(str(value) for value in customer_ids)
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()[:16]


def _rank_distribution(values):
    counts = values.value_counts().reindex(RANK_ORDER, fill_value=0)
    total = int(counts.sum())
    return {
        rank: {
            "count": int(counts[rank]),
            "percent": float(counts[rank] / total * 100) if total else 0.0,
        }
        for rank in RANK_ORDER
    }


def _attr_iso(dataset, key):
    value = dataset.attrs.get(key)
    return value.isoformat() if hasattr(value, "isoformat") else value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train Random Forest for the four CustomerRank classes."
    )
    parser.add_argument("--output-dir", required=True, help="New run folder inside outputs/runs.")
    parser.add_argument("--data", default=None, help="Path to Online Retail.xlsx.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--min-samples-split", type=int, default=DEFAULT_MIN_SAMPLES_SPLIT)
    parser.add_argument("--min-samples-leaf", type=int, default=DEFAULT_MIN_SAMPLES_LEAF)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if output_dir.exists():
        parser.error(f"output directory already exists; choose a new directory: {output_dir}")
    safe_root = (PROJECT_ROOT / "outputs" / "runs").resolve()
    try:
        output_dir.resolve().relative_to(safe_root)
    except ValueError:
        parser.error(f"output directory must be a new run folder inside {safe_root}")

    result = run_training(
        data_loader.load_data(args.data),
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=args.n_jobs,
    )
    saved_to = save_experiment(result, output_dir)
    print(f"Completed Random Forest rank evaluation: {saved_to}")
    print(pd.DataFrame([result["result_row"]]).to_string(index=False))


if __name__ == "__main__":
    main()
