"""Multi-cutoff, fixed-configuration backtest for Gradient Boosting ranks.

This script does not modify the production pipeline. Each cutoff uses only
pre-cutoff transactions for features, a complete following 90-day target
window, and the existing customer-level split and train-only rank thresholds.

Example:
    python -m src.backtest_gradient_boosting \
        --output-dir outputs/experiments/gradient_boosting_multicutoff_20261105-01
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight

from src.data_loader import load_data
from src.evaluate import COMMON_METRICS, evaluate_classification
from src.feature_engineering import FEATURE_COLUMNS, RANK_ORDER, split_customer_dataset
from src.predict import build_model_input
from src.preprocessing import clean_transactions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_DAYS = 90
TEST_SIZE = 0.2
RANDOM_STATE = 42
CUTOFFS = (
    pd.Timestamp("2011-03-02"),
    pd.Timestamp("2011-06-01"),
    pd.Timestamp("2011-09-01"),
)
CONFIGURATIONS = {
    "Baseline": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "random_state": RANDOM_STATE,
        "sample_weight": False,
    },
    "Balanced": {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_samples_split": 20,
        "min_samples_leaf": 5,
        "random_state": RANDOM_STATE,
        "sample_weight": True,
    },
}


def _evaluation_set_id(customer_ids):
    normalized = sorted(str(value) for value in customer_ids)
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()[:16]


def _build_dataset_at_cutoff(transactions: pd.DataFrame, cutoff: pd.Timestamp):
    """Build the shared six historical features and existing forward label target."""
    prediction_end = cutoff + pd.Timedelta(days=PREDICTION_DAYS)
    history = transactions.loc[transactions["InvoiceDate"] < cutoff]
    if history.empty:
        raise ValueError(f"No history is available before cutoff {cutoff}.")

    grouped = history.groupby("CustomerID", sort=True)
    last_purchase = grouped["InvoiceDate"].max()
    frequency = grouped["InvoiceNo"].nunique()
    monetary = grouped["TotalPrice"].sum()
    total_quantity = grouped["Quantity"].sum()
    unique_products = grouped["StockCode"].nunique()
    dataset = pd.DataFrame(
        {
            "CustomerID": last_purchase.index,
            "Recency": (cutoff - last_purchase).dt.days.astype(int).to_numpy(),
            "Frequency": frequency.to_numpy(),
            "Monetary": monetary.to_numpy(),
            "TotalQuantity": total_quantity.to_numpy(),
            "UniqueProducts": unique_products.to_numpy(),
        }
    )
    dataset["AvgOrderValue"] = dataset["Monetary"] / dataset["Frequency"]
    dataset = dataset[["CustomerID", *FEATURE_COLUMNS]]
    history_ids = set(dataset["CustomerID"])
    future = transactions.loc[
        (transactions["InvoiceDate"] >= cutoff)
        & (transactions["InvoiceDate"] <= prediction_end)
        & transactions["CustomerID"].isin(history_ids)
    ]
    future_spend = future.groupby("CustomerID", sort=True)["TotalPrice"].sum()
    dataset["FutureSpend"] = dataset["CustomerID"].map(future_spend).fillna(0.0).astype(float)
    dataset["ReturnedNext90Days"] = (dataset["FutureSpend"] > 0).astype(int)
    dataset.attrs["prediction_start"] = cutoff
    dataset.attrs["observation_end"] = prediction_end
    return dataset, len(history), len(future), prediction_end


def _score(y_true, y_pred):
    metrics = evaluate_classification(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(RANK_ORDER), zero_division=0
    )
    result = {key: float(metrics[key]) for key in COMMON_METRICS}
    for rank, p, r, score, count in zip(RANK_ORDER, precision, recall, f1, support):
        result[f"{rank}_Precision"] = float(p)
        result[f"{rank}_Recall"] = float(r)
        result[f"{rank}_F1"] = float(score)
        result[f"{rank}_Support"] = int(count)
    result["Confusion_Matrix"] = confusion_matrix(y_true, y_pred, labels=list(RANK_ORDER)).tolist()
    result["Missing_True_Classes"] = metrics["missing_true_classes"]
    result["Missing_Predicted_Classes"] = metrics["missing_predicted_classes"]
    return result


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def run_backtest(transactions: pd.DataFrame, *, output_dir: Path | str):
    if not isinstance(transactions, pd.DataFrame) or transactions.empty:
        raise ValueError("transactions must be a non-empty DataFrame.")
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    if output_path.exists():
        raise FileExistsError(f"Refusing to write into an existing experiment directory: {output_path}")
    experiments_root = (PROJECT_ROOT / "outputs" / "experiments").resolve()
    try:
        output_path.resolve().relative_to(experiments_root)
    except ValueError as exc:
        raise ValueError(f"Backtest output must be inside {experiments_root}") from exc

    clean = clean_transactions(transactions)
    data_start = clean["InvoiceDate"].min()
    data_end = clean["InvoiceDate"].max()
    cutoff_checks = []
    for cutoff in CUTOFFS:
        prediction_end = cutoff + pd.Timedelta(days=PREDICTION_DAYS)
        history_days = (cutoff - data_start).total_seconds() / 86400
        if history_days < PREDICTION_DAYS:
            raise ValueError(f"Cutoff {cutoff.date()} has less than 90 days of historical coverage.")
        if prediction_end > data_end:
            raise ValueError(
                f"Cutoff {cutoff.date()} lacks a complete 90-day future window; data ends at {data_end}."
            )
        cutoff_checks.append(
            {
                "cutoff": cutoff,
                "prediction_end": prediction_end,
                "historical_coverage_days": history_days,
            }
        )

    cutoff_rows = []
    prediction_rows = []
    per_cutoff = {}
    thresholds_by_cutoff = {}
    class_distributions = {}
    test_customer_ids_by_cutoff = {}

    for check in cutoff_checks:
        cutoff = check["cutoff"]
        dataset, history_rows, future_rows, prediction_end = _build_dataset_at_cutoff(clean, cutoff)
        train, test, thresholds = split_customer_dataset(
            dataset, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        train_ids = set(train["CustomerID"])
        test_ids = set(test["CustomerID"])
        if train_ids & test_ids:
            raise RuntimeError(f"CustomerID leakage detected at cutoff {cutoff.date()}.")
        if train.empty or test.empty:
            raise ValueError(f"Cutoff {cutoff.date()} has an empty train or test split.")

        X_train = build_model_input(train)
        X_test = build_model_input(test)
        y_train = train["CustomerRank"].to_numpy()
        y_test = test["CustomerRank"].to_numpy()
        evaluation_set_id = _evaluation_set_id(test["CustomerID"])
        distribution = {
            split_name: {
                rank: int(frame["CustomerRank"].value_counts().get(rank, 0))
                for rank in RANK_ORDER
            }
            for split_name, frame in (("train", train), ("test", test))
        }
        class_distributions[cutoff.date().isoformat()] = distribution
        thresholds_by_cutoff[cutoff.date().isoformat()] = thresholds.as_dict()
        test_customer_ids_by_cutoff[cutoff.date().isoformat()] = test_ids
        per_cutoff[cutoff.date().isoformat()] = {}
        print(
            f"Cutoff {cutoff.date()} (target through {prediction_end.date()}): "
            f"customers train={len(train)}, test={len(test)}, eval={evaluation_set_id}",
            flush=True,
        )

        for config_name, config in CONFIGURATIONS.items():
            params = {key: value for key, value in config.items() if key != "sample_weight"}
            model = GradientBoostingClassifier(**params)
            sample_weights = (
                compute_sample_weight(class_weight="balanced", y=y_train)
                if config["sample_weight"]
                else None
            )
            model.fit(X_train, y_train, sample_weight=sample_weights)
            predictions = model.predict(X_test)
            scores = _score(y_test, predictions)
            per_cutoff[cutoff.date().isoformat()][config_name] = scores
            row = {
                "Cutoff": cutoff.date().isoformat(),
                "Prediction_End": prediction_end.date().isoformat(),
                "Configuration": config_name,
                "Evaluation_Set_ID": evaluation_set_id,
                "Historical_Customer_Count": len(dataset),
                "History_Transaction_Rows": history_rows,
                "Future_Transaction_Rows": future_rows,
                "Train_Customers": len(train),
                "Test_Customers": len(test),
                "Rank_Threshold_q50": thresholds.q50,
                "Rank_Threshold_q85": thresholds.q85,
                "Train_Rank_Distribution": json.dumps(distribution["train"]),
                "Test_Rank_Distribution": json.dumps(distribution["test"]),
                "Sample_Weight": "balanced" if config["sample_weight"] else "none",
                **{key: value for key, value in config.items() if key != "sample_weight"},
                **{key: value for key, value in scores.items() if key not in {"Confusion_Matrix", "Missing_True_Classes", "Missing_Predicted_Classes"}},
                "Confusion_Matrix": json.dumps(scores["Confusion_Matrix"]),
                "Missing_True_Classes": json.dumps(scores["Missing_True_Classes"]),
                "Missing_Predicted_Classes": json.dumps(scores["Missing_Predicted_Classes"]),
            }
            cutoff_rows.append(row)
            for customer_id, actual, predicted in zip(test["CustomerID"], y_test, predictions):
                prediction_rows.append(
                    {
                        "Cutoff": cutoff.date().isoformat(),
                        "Configuration": config_name,
                        "Evaluation_Set_ID": evaluation_set_id,
                        "CustomerID": customer_id,
                        "CustomerRank_true": actual,
                        "CustomerRank_predicted": predicted,
                    }
                )
            print(
                f"  {config_name}: Macro F1={scores['F1_Macro']:.4f}, "
                f"Balanced Accuracy={scores['Balanced_Accuracy']:.4f}",
                flush=True,
            )

    per_cutoff_results = pd.DataFrame(cutoff_rows)
    aggregate_rows = []
    numeric_metrics = [
        *COMMON_METRICS,
        *[f"{rank}_{metric}" for rank in RANK_ORDER for metric in ("Precision", "Recall", "F1")],
    ]
    for config_name in CONFIGURATIONS:
        subset = per_cutoff_results.loc[per_cutoff_results["Configuration"] == config_name]
        aggregate = {"Configuration": config_name, "Cutoff_Count": len(subset)}
        for metric in numeric_metrics:
            aggregate[f"{metric}_Mean"] = float(subset[metric].mean())
            aggregate[f"{metric}_Std"] = float(subset[metric].std(ddof=1)) if len(subset) > 1 else 0.0
        aggregate_rows.append(aggregate)
    aggregate_results = pd.DataFrame(aggregate_rows)
    macro_by_cutoff = per_cutoff_results.pivot(index="Cutoff", columns="Configuration", values="F1_Macro")
    balanced_wins = int((macro_by_cutoff["Balanced"] > macro_by_cutoff["Baseline"]).sum())
    aggregate_results["Balanced_Better_MacroF1_Cutoffs"] = balanced_wins
    aggregate_results["Balanced_Better_BalancedAccuracy_Cutoffs"] = int(
        (
            per_cutoff_results.pivot(index="Cutoff", columns="Configuration", values="Balanced_Accuracy")["Balanced"]
            > per_cutoff_results.pivot(index="Cutoff", columns="Configuration", values="Balanced_Accuracy")["Baseline"]
        ).sum()
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.mkdir()
    per_cutoff_results.to_csv(output_path / "backtest_by_cutoff.csv", index=False, encoding="utf-8", mode="x")
    aggregate_results.to_csv(output_path / "aggregate_summary.csv", index=False, encoding="utf-8", mode="x")
    pd.DataFrame(prediction_rows).to_csv(
        output_path / "predictions_by_cutoff.csv", index=False, encoding="utf-8", mode="x"
    )
    with (output_path / "confusion_matrices.json").open("x", encoding="utf-8") as stream:
        json.dump({"labels": list(RANK_ORDER), "per_cutoff": per_cutoff}, stream, ensure_ascii=False, indent=2, default=_json_default)

    metadata = {
        "experiment": "fixed-configuration multi-cutoff Gradient Boosting backtest",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_data_min_valid_transaction": data_start.isoformat(),
        "source_data_max_valid_transaction": data_end.isoformat(),
        "valid_transaction_rows": len(clean),
        "cutoffs": [
            {
                "cutoff": check["cutoff"].isoformat(),
                "prediction_end": check["prediction_end"].isoformat(),
                "historical_coverage_days": check["historical_coverage_days"],
            }
            for check in cutoff_checks
        ],
        "prediction_days": PREDICTION_DAYS,
        "test_size": TEST_SIZE,
        "split_random_state": RANDOM_STATE,
        "split_rule": "existing split_customer_dataset; customer-level random split, independently within each cutoff",
        "threshold_rule": "existing positive train FutureSpend q50/q85; test labels use these train thresholds",
        "feature_columns": list(FEATURE_COLUMNS),
        "features_use_transactions_before_cutoff_only": True,
        "future_labels_use_cutoff_to_cutoff_plus_90_days": True,
        "all_cutoff_test_set_ids": {
            cutoff: sorted(set(
                per_cutoff_results.loc[
                    per_cutoff_results["Cutoff"] == cutoff, "Evaluation_Set_ID"
                ].astype(str)
            ))[0]
            for cutoff in per_cutoff_results["Cutoff"].unique()
        },
        "rank_thresholds_by_cutoff": thresholds_by_cutoff,
        "rank_distributions_by_cutoff": class_distributions,
        "configurations_fixed_before_backtest": CONFIGURATIONS,
        "config_selection_from_cutoff_test_results": False,
        "balanced_better_macro_f1_cutoffs": balanced_wins,
        "production_pipeline_changed": False,
        "sklearn_version": sklearn.__version__,
        "test_customer_ids_overlap_across_cutoffs": {
            f"{left}|{right}": len(test_customer_ids_by_cutoff[left] & test_customer_ids_by_cutoff[right])
            for i, left in enumerate(test_customer_ids_by_cutoff)
            for right in list(test_customer_ids_by_cutoff)[i + 1 :]
        },
    }
    with (output_path / "backtest_metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, default=_json_default)
    print("Aggregate results:")
    print(aggregate_results.to_string(index=False))
    print(f"Backtest artifacts saved to {output_path}")
    return {
        "per_cutoff": per_cutoff_results,
        "aggregate": aggregate_results,
        "metadata": metadata,
        "output_dir": output_path,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Backtest two fixed Gradient Boosting settings across multiple cutoffs.")
    parser.add_argument(
        "--output-dir", required=True,
        help="New directory under outputs/experiments; existing directories are protected.",
    )
    parser.add_argument("--data", default=None, help="Path to Online Retail.xlsx; defaults to the project raw file.")
    args = parser.parse_args(argv)
    return run_backtest(load_data(args.data), output_dir=args.output_dir)


if __name__ == "__main__":
    main()
