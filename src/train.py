"""Leakage-safe customer rank training entry point (K-Means integration first).

Example, using an output directory that does not already exist:
    python -m src.train --output-dir outputs/runs/rank-kmeans-run-01

The command reads the configured raw dataset but writes only to the explicitly
provided, new run directory. It never replaces the legacy K-Means artifacts.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import load_data
from src.evaluate import evaluate_classification, evaluate_regression, make_model_result, write_model_results
from src.feature_engineering import FEATURE_COLUMNS, build_customer_dataset, split_customer_dataset
from src.models.kmeans_model import KMeansRankClassifier
from src.models.linear_regression_model import LinearRegressionRankModel
from src.preprocessing import clean_transactions


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train_kmeans_rank_model(
    dataset: pd.DataFrame,
    *,
    n_clusters: int = 4,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Fit K-Means on train rows and evaluate the mapped four ranks on test.

    Thresholds, scaler, cluster centroids, and the RFM business-order mapping
    are learned from the customer-level training subset only. CustomerRank is
    not passed into K-Means or its mapper; test labels are used only for metrics.
    """
    if not isinstance(dataset, pd.DataFrame) or dataset.empty:
        raise ValueError("dataset must be a non-empty customer DataFrame.")
    required_targets = {"FutureSpend", "CustomerRank", "ReturnedNext90Days"}
    if not required_targets.issubset(set(dataset.columns)):
        missing_targets = sorted(required_targets - set(dataset.columns))
        raise KeyError(f"Missing target columns: {', '.join(missing_targets)}")
    missing_features = [column for column in FEATURE_COLUMNS if column not in dataset.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns: {', '.join(missing_features)}")
    if dataset["CustomerID"].isna().any() or dataset["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must be present and unique before model training.")

    train, test, thresholds = split_customer_dataset(dataset, test_size=test_size, random_state=random_state)
    if len(train) < n_clusters:
        raise ValueError(f"Training split has {len(train)} customers, fewer than {n_clusters} clusters.")

    model = KMeansRankClassifier(n_clusters=n_clusters, random_state=random_state)
    started = time.perf_counter()
    # K-Means and its business-value ordering are unsupervised: CustomerRank
    # labels are intentionally not passed to fit or cluster mapping.
    model.fit(train[list(FEATURE_COLUMNS)])
    training_time = time.perf_counter() - started

    evaluation_set_id = _evaluation_set_id(test["CustomerID"])
    if test.empty:
        metrics = None
        prediction_time = None
        predictions = pd.DataFrame(columns=["CustomerID", "CustomerRank_true", "CustomerRank_predicted", "Cluster"])
        result_row = make_model_result(
            "K-Means", "Clustering", status="no_test_samples", training_time=training_time,
            evaluation_set_id=evaluation_set_id,
        )
        diagnostics = model.cluster_diagnostics()
    else:
        started = time.perf_counter()
        output = model.predict_with_status(test[list(FEATURE_COLUMNS)])
        prediction_time = time.perf_counter() - started
        if output["unmapped_clusters"]:
            raise ValueError(
                "The fitted K-Means model predicted cluster IDs without a train-derived rank mapping: "
                f"{output['unmapped_clusters']}"
            )
        metrics = evaluate_classification(test["CustomerRank"], output["ranks"])
        result_row = make_model_result(
            "K-Means",
            "Clustering",
            metrics,
            training_time=training_time,
            prediction_time=prediction_time,
            evaluation_set_id=evaluation_set_id,
        )
        predictions = pd.DataFrame(
            {
                "CustomerID": test["CustomerID"].to_numpy(),
                "CustomerRank_true": test["CustomerRank"].to_numpy(),
                "CustomerRank_predicted": output["ranks"],
                "Cluster": output["clusters"],
            }
        )
        diagnostics = model.cluster_diagnostics()

    train_ids = train["CustomerID"].tolist()
    test_ids = test["CustomerID"].tolist()
    if set(train_ids) & set(test_ids):
        raise RuntimeError("Customer ID leakage detected between train and test.")
    metadata = {
        "model": "K-Means",
        "model_type": "Clustering",
        "random_state": random_state,
        "test_size": test_size,
        "n_clusters": n_clusters,
        "n_train_customers": len(train),
        "n_test_customers": len(test),
        "evaluation_set_id": evaluation_set_id,
        "rank_thresholds": thresholds.as_dict(),
        "cluster_to_rank": dict(model.rank_mapper_.cluster_to_rank_),
        "cluster_mapping_strategy": model.rank_mapper_.mapping_strategy_,
        "cluster_mapping_formula": model.rank_mapper_.mapping_formula_,
        "cluster_profiles_train": model.rank_mapper_.cluster_profiles_.to_dict(orient="records"),
        "unmapped_cluster_policy": "raise; never infer a rank from an unknown ClusterID",
        "train_only_clustering_diagnostics": diagnostics,
        "train_rank_distribution": _rank_distribution(train["CustomerRank"]),
        "test_rank_distribution": _rank_distribution(test["CustomerRank"]),
        "prediction_unmapped_clusters": [] if test.empty else output["unmapped_clusters"],
        "prediction_window_start": _attr_iso(dataset, "prediction_start"),
        "observation_end": _attr_iso(dataset, "observation_end"),
        "temporal_scope": "single cutoff; not a multi-period backtest",
    }
    return {
        "model": model,
        "metrics": metrics,
        "result_row": result_row,
        "predictions": predictions,
        "metadata": metadata,
        "train_customer_ids": train_ids,
        "test_customer_ids": test_ids,
        "thresholds": thresholds,
        "cluster_diagnostics": diagnostics,
    }


def train_linear_regression_rank_model(
    dataset: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Fit Linear Regression on train rows and evaluate on test."""
    if not isinstance(dataset, pd.DataFrame) or dataset.empty:
        raise ValueError("dataset must be a non-empty customer DataFrame.")
    required_targets = {"FutureSpend", "CustomerRank", "ReturnedNext90Days"}
    if not required_targets.issubset(set(dataset.columns)):
        missing_targets = sorted(required_targets - set(dataset.columns))
        raise KeyError(f"Missing target columns: {', '.join(missing_targets)}")
    missing_features = [column for column in FEATURE_COLUMNS if column not in dataset.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns: {', '.join(missing_features)}")
    if dataset["CustomerID"].isna().any() or dataset["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must be present and unique before model training.")

    train, test, thresholds = split_customer_dataset(dataset, test_size=test_size, random_state=random_state)

    model = LinearRegressionRankModel()
    started = time.perf_counter()
    model.fit(train[list(FEATURE_COLUMNS)], train["FutureSpend"], thresholds=thresholds)
    training_time = time.perf_counter() - started

    evaluation_set_id = _evaluation_set_id(test["CustomerID"])
    if test.empty:
        metrics = None
        prediction_time = None
        predictions = pd.DataFrame(
            columns=["CustomerID", "CustomerRank_true", "CustomerRank_predicted", "Cluster", "Spend_predicted", "Spend_true"]
        )
        result_row = make_model_result(
            "Linear Regression",
            "Regression",
            status="no_test_samples",
            training_time=training_time,
            evaluation_set_id=evaluation_set_id,
        )
    else:
        started = time.perf_counter()
        spend_predictions = model.predict(test[list(FEATURE_COLUMNS)])
        prediction_time = time.perf_counter() - started
        rank_predictions = model.predict_rank(test[list(FEATURE_COLUMNS)], thresholds=thresholds)

        metrics = evaluate_regression(test["FutureSpend"], spend_predictions, thresholds)
        result_row = make_model_result(
            "Linear Regression",
            "Regression",
            metrics,
            training_time=training_time,
            prediction_time=prediction_time,
            evaluation_set_id=evaluation_set_id,
        )
        predictions = pd.DataFrame(
            {
                "CustomerID": test["CustomerID"].to_numpy(),
                "CustomerRank_true": test["CustomerRank"].to_numpy(),
                "CustomerRank_predicted": rank_predictions,
                "Cluster": 0,
                "Spend_predicted": spend_predictions,
                "Spend_true": test["FutureSpend"].to_numpy(),
            }
        )

    train_ids = train["CustomerID"].tolist()
    test_ids = test["CustomerID"].tolist()
    if set(train_ids) & set(test_ids):
        raise RuntimeError("Customer ID leakage detected between train and test.")

    metadata = {
        "model": "Linear Regression",
        "model_type": "Regression",
        "random_state": random_state,
        "test_size": test_size,
        "n_train_customers": len(train),
        "n_test_customers": len(test),
        "evaluation_set_id": evaluation_set_id,
        "rank_thresholds": thresholds.as_dict(),
        "coefficients": model.get_coefficients().to_dict(orient="records"),
        "train_rank_distribution": _rank_distribution(train["CustomerRank"]),
        "test_rank_distribution": _rank_distribution(test["CustomerRank"]),
        "prediction_window_start": _attr_iso(dataset, "prediction_start"),
        "observation_end": _attr_iso(dataset, "observation_end"),
        "temporal_scope": "single cutoff; not a multi-period backtest",
    }
    return {
        "model": model,
        "metrics": metrics,
        "result_row": result_row,
        "predictions": predictions,
        "metadata": metadata,
        "train_customer_ids": train_ids,
        "test_customer_ids": test_ids,
        "thresholds": thresholds,
    }


def save_experiment(result: dict, output_dir: Path | str):
    """Write a complete run to a new directory, refusing every overwrite."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to write into an existing run directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    joblib.dump(result["model"], destination / "model.joblib")
    write_model_results([result["result_row"]], destination / "ranking_metrics.csv")
    result["predictions"].to_csv(destination / "test_predictions.csv", index=False, encoding="utf-8")
    with (destination / "run_metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(result["metadata"], stream, ensure_ascii=False, indent=2)
    return destination


def run_training(
    transactions: pd.DataFrame,
    *,
    model_name: str = "kmeans",
    prediction_days: int = 90,
    n_clusters: int = 4,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """In-memory training path, useful for tests and notebooks."""
    clean = clean_transactions(transactions)
    dataset = build_customer_dataset(clean, prediction_days=prediction_days, random_state=random_state)
    norm_name = str(model_name).strip().lower()
    if norm_name in ("linear_regression", "linear regression", "lr", "regression"):
        result = train_linear_regression_rank_model(
            dataset, test_size=test_size, random_state=random_state
        )
    else:
        result = train_kmeans_rank_model(
            dataset, n_clusters=n_clusters, test_size=test_size, random_state=random_state
        )
    result["dataset"] = dataset
    return result


def _evaluation_set_id(customer_ids):
    normalized = sorted(str(value) for value in customer_ids)
    digest = hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()
    return digest[:16]


def _attr_iso(dataset, key):
    value = dataset.attrs.get(key)
    return value.isoformat() if hasattr(value, "isoformat") else value


def _rank_distribution(values):
    rank_order = ("Diamond", "Gold", "Silver", "Bronze")
    counts = values.value_counts().reindex(rank_order, fill_value=0)
    total = int(counts.sum())
    return {
        rank: {
            "count": int(counts[rank]),
            "percent": float(counts[rank] / total * 100) if total else 0.0,
        }
        for rank in rank_order
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate K-Means for the four CustomerRank classes.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New run folder that does not exist yet; existing artifacts are protected.",
    )
    parser.add_argument("--data", default=None, help="Path to Online Retail.xlsx (default: data/raw/Online Retail.xlsx).")
    parser.add_argument("--model", default="kmeans", choices=["kmeans", "linear_regression", "lr"], help="Model name to train")
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if output_dir.exists():
        parser.error(f"output directory already exists; choose a new directory: {output_dir}")
    safe_root = (PROJECT_ROOT / "outputs" / "runs").resolve()
    resolved_output = output_dir.resolve()
    try:
        resolved_output.relative_to(safe_root)
    except ValueError:
        parser.error(f"output directory must be a new run folder inside {safe_root}")
    transactions = load_data(args.data)
    result = run_training(
        transactions,
        model_name=args.model,
        n_clusters=args.clusters,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    saved_to = save_experiment(result, output_dir)
    print(f"Completed K-Means rank evaluation: {saved_to}")
    print(pd.DataFrame([result["result_row"]]).to_string(index=False))
    for split_name in ("train", "test"):
        distribution = result["metadata"][f"{split_name}_rank_distribution"]
        summary = " ".join(
            f"{rank}={distribution[rank]['count']} ({distribution[rank]['percent']:.2f}%)"
            for rank in ("Diamond", "Gold", "Silver", "Bronze")
        )
        print(f"{split_name.title()} CustomerRank: {summary}")
    print("Silhouette/Inertia are train-only clustering diagnostics; this run uses one temporal cutoff.")


if __name__ == "__main__":
    main()
