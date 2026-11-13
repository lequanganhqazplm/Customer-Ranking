"""Fixed-window, multi-cutoff benchmark for Customer-Ranking classifiers.

The script is intentionally separate from the production pipelines. For every
cutoff it builds features from [cutoff - 90 days, cutoff), labels from
[cutoff, cutoff + 90 days), validates all splits before fitting any estimator,
and writes results only to a new directory under outputs/experiments/.
"""

import argparse
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from src.data_loader import load_data
from src.feature_engineering import (
    FEATURE_COLUMNS,
    RANK_ORDER,
    assign_customer_rank,
    fit_rank_thresholds,
)
from src.models.decision_tree_model import DecisionTreeRankClassifier
from src.models.gradient_boosting_model import GradientBoostingRankClassifier
from src.models.random_forest_model import RandomForestRankClassifier
from src.preprocessing import clean_transactions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = (PROJECT_ROOT / "outputs" / "experiments").resolve()
HISTORY_DAYS = 90
FUTURE_DAYS = 90
TEST_SIZE = 0.2
RANDOM_STATE = 42
CUTOFFS = (
    pd.Timestamp("2011-03-02 00:00:00"),
    pd.Timestamp("2011-06-01 00:00:00"),
    pd.Timestamp("2011-09-01 00:00:00"),
)

MODEL_CONFIGURATIONS = {
    "Decision Tree": {
        "max_depth": 5,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "random_state": RANDOM_STATE,
    },
    "Random Forest": {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "bootstrap": True,
        "class_weight": None,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "Gradient Boosting Baseline": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_samples_split": 20,
        "min_samples_leaf": 10,
        "random_state": RANDOM_STATE,
        "sample_weight": None,
    },
    "Gradient Boosting Balanced": {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_samples_split": 20,
        "min_samples_leaf": 5,
        "random_state": RANDOM_STATE,
        "sample_weight": "compute_sample_weight(class_weight='balanced', y=y_train)",
    },
}


def _canonical_customer_id(value):
    number = float(value)
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"CustomerID must be a finite whole number: {value!r}")
    return str(int(number))


def _evaluation_set_id(customer_ids):
    normalized = sorted(_canonical_customer_id(value) for value in customer_ids)
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()[:16]


def _class_distribution(labels):
    counts = pd.Series(labels).value_counts().reindex(RANK_ORDER, fill_value=0)
    return {rank: int(counts[rank]) for rank in RANK_ORDER}


def _build_cutoff_dataset(transactions, cutoff):
    """Build independent features/targets using exact half-open time windows."""
    cutoff = pd.Timestamp(cutoff)
    history_start = cutoff - pd.Timedelta(days=HISTORY_DAYS)
    future_end = cutoff + pd.Timedelta(days=FUTURE_DAYS)

    history_mask = transactions["InvoiceDate"].ge(history_start) & transactions["InvoiceDate"].lt(cutoff)
    future_mask = transactions["InvoiceDate"].ge(cutoff) & transactions["InvoiceDate"].lt(future_end)
    history = transactions.loc[history_mask].copy()
    future_window = transactions.loc[future_mask].copy()
    if history.empty:
        raise ValueError(f"No valid transactions in 90-day history window for {cutoff.date()}.")
    if future_window.empty:
        raise ValueError(f"No valid transactions in 90-day future window for {cutoff.date()}.")

    # Explicit timestamp assertions guard against an inclusive upper endpoint.
    if not history["InvoiceDate"].ge(history_start).all() or not history["InvoiceDate"].lt(cutoff).all():
        raise RuntimeError(f"History window boundary validation failed at {cutoff.date()}.")
    if not future_window["InvoiceDate"].ge(cutoff).all() or not future_window["InvoiceDate"].lt(future_end).all():
        raise RuntimeError(f"Future window boundary validation failed at {cutoff.date()}.")

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
            "Monetary": monetary.to_numpy(dtype=float),
            "TotalQuantity": total_quantity.to_numpy(dtype=float),
            "UniqueProducts": unique_products.to_numpy(dtype=int),
        }
    )
    dataset["AvgOrderValue"] = dataset["Monetary"] / dataset["Frequency"]
    dataset = dataset[["CustomerID", *FEATURE_COLUMNS]]

    cohort_ids = set(dataset["CustomerID"])
    future_window = future_window.loc[future_window["CustomerID"].isin(cohort_ids)]
    future_spend = future_window.groupby("CustomerID", sort=True)["TotalPrice"].sum()
    dataset["FutureSpend"] = dataset["CustomerID"].map(future_spend).fillna(0.0).astype(float)
    dataset["ReturnedNext90Days"] = (dataset["FutureSpend"] > 0).astype(int)

    return dataset, {
        "cutoff": cutoff,
        "history_start": history_start,
        "history_end_exclusive": cutoff,
        "future_start_inclusive": cutoff,
        "future_end_exclusive": future_end,
        "history_transaction_rows": int(len(history)),
        "future_transaction_rows_for_cohort": int(len(future_window)),
        "history_observed_min": history["InvoiceDate"].min(),
        "history_observed_max": history["InvoiceDate"].max(),
        "future_observed_min": future_window["InvoiceDate"].min() if not future_window.empty else None,
        "future_observed_max": future_window["InvoiceDate"].max() if not future_window.empty else None,
        "history_at_lower_boundary": int(history["InvoiceDate"].eq(history_start).sum()),
        "history_at_cutoff_excluded": int(transactions["InvoiceDate"].eq(cutoff).sum()),
        "future_at_cutoff_included": int(future_window["InvoiceDate"].eq(cutoff).sum()),
        "future_at_upper_boundary_excluded": int(transactions["InvoiceDate"].eq(future_end).sum()),
    }


def _prepare_and_validate(transactions):
    clean = clean_transactions(transactions)
    if clean.empty:
        raise ValueError("No valid transactions remain after preprocessing.")
    plans = []
    validation = {
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "valid_transaction_source": True,
            "history_window_is_half_open_90_days": True,
            "future_window_is_half_open_90_days": True,
            "features_rebuilt_independently_at_each_cutoff": True,
            "model_features_are_exactly_the_six_required_columns": True,
            "target_and_identifier_columns_excluded_from_model_input": True,
            "customer_id_train_test_overlap": False,
            "q50_q85_fit_only_on_train_future_spend": True,
            "same_split_features_labels_and_evaluation_id_across_models": True,
            "test_used_for_tuning_or_selection": False,
        },
        "source_data": {
            "valid_transaction_rows": int(len(clean)),
            "min_valid_timestamp": clean["InvoiceDate"].min().isoformat(),
            "max_valid_timestamp": clean["InvoiceDate"].max().isoformat(),
        },
        "feature_columns": list(FEATURE_COLUMNS),
        "cutoffs": {},
    }

    for cutoff in CUTOFFS:
        dataset, windows = _build_cutoff_dataset(clean, cutoff)
        if dataset["CustomerID"].isna().any() or dataset["CustomerID"].duplicated().any():
            raise ValueError(f"Cohort CustomerID is missing or duplicated at {cutoff.date()}.")
        if dataset[list(FEATURE_COLUMNS)].isna().any().any() or not np.isfinite(
            dataset[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        ).all():
            raise ValueError(f"Features contain missing or non-finite values at {cutoff.date()}.")

        positions = np.arange(len(dataset))
        from sklearn.model_selection import train_test_split

        train_positions, test_positions = train_test_split(
            positions, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
        )
        train = dataset.iloc[train_positions].copy()
        test = dataset.iloc[test_positions].copy()
        if set(train["CustomerID"]) & set(test["CustomerID"]):
            raise ValueError(f"Train/test CustomerID overlap at {cutoff.date()}.")
        if train.empty or test.empty:
            raise ValueError(f"Empty train or test split at {cutoff.date()}.")

        thresholds = fit_rank_thresholds(train["FutureSpend"])
        if thresholds.q50 < 0 or thresholds.q85 < thresholds.q50:
            raise ValueError(f"Invalid train-only rank thresholds at {cutoff.date()}.")
        train["CustomerRank"] = assign_customer_rank(train["FutureSpend"], thresholds.q50, thresholds.q85)
        test["CustomerRank"] = assign_customer_rank(test["FutureSpend"], thresholds.q50, thresholds.q85)
        if set(train["CustomerRank"]) != set(RANK_ORDER):
            raise ValueError(f"Training split lacks one or more rank classes at {cutoff.date()}.")

        # Select features only after preserving labels/targets separately.
        X_train = train.loc[:, list(FEATURE_COLUMNS)].copy()
        X_test = test.loc[:, list(FEATURE_COLUMNS)].copy()
        forbidden = {"CustomerID", "FutureSpend", "CustomerRank", "ReturnedNext90Days"}
        if set(X_train.columns) != set(FEATURE_COLUMNS) or set(X_test.columns) != set(FEATURE_COLUMNS):
            raise ValueError(f"Feature columns mismatch at {cutoff.date()}.")
        if forbidden & set(X_train.columns) or forbidden & set(X_test.columns):
            raise ValueError(f"Target/identifier leakage in model features at {cutoff.date()}.")
        if not np.array_equal(X_train.columns.to_numpy(), np.asarray(FEATURE_COLUMNS)):
            raise ValueError(f"Feature order mismatch at {cutoff.date()}.")

        eval_id = _evaluation_set_id(test["CustomerID"])
        cutoff_validation = {
            **{key: value.isoformat() if isinstance(value, pd.Timestamp) else value for key, value in windows.items()},
            "cohort_customers": int(len(dataset)),
            "train_customers": int(len(train)),
            "test_customers": int(len(test)),
            "train_test_customer_overlap": 0,
            "train_customer_ids": sorted(_canonical_customer_id(value) for value in train["CustomerID"]),
            "test_customer_ids": sorted(_canonical_customer_id(value) for value in test["CustomerID"]),
            "evaluation_set_id": eval_id,
            "rank_thresholds_train_only": thresholds.as_dict(),
            "train_rank_distribution": _class_distribution(train["CustomerRank"]),
            "test_rank_distribution": _class_distribution(test["CustomerRank"]),
            "feature_columns": list(X_train.columns),
            "forbidden_feature_columns_present": sorted(forbidden & set(X_train.columns)),
            "checks_passed": True,
        }
        validation["cutoffs"][cutoff.date().isoformat()] = cutoff_validation
        plans.append(
            {
                "cutoff": cutoff,
                "dataset": dataset,
                "train": train,
                "test": test,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": train["CustomerRank"].to_numpy(),
                "y_test": test["CustomerRank"].to_numpy(),
                "thresholds": thresholds,
                "evaluation_set_id": eval_id,
                "window_metadata": windows,
            }
        )

    return clean, plans, validation


def _metric_row(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(RANK_ORDER), zero_division=0
    )
    result = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro_Precision": float(precision_score(y_true, y_pred, labels=list(RANK_ORDER), average="macro", zero_division=0)),
        "Macro_Recall": float(recall_score(y_true, y_pred, labels=list(RANK_ORDER), average="macro", zero_division=0)),
        "Macro_F1": float(f1_score(y_true, y_pred, labels=list(RANK_ORDER), average="macro", zero_division=0)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Confusion_Matrix": confusion_matrix(y_true, y_pred, labels=list(RANK_ORDER)).tolist(),
    }
    for rank, p, r, score, count in zip(RANK_ORDER, precision, recall, f1, support):
        result[f"{rank}_Precision"] = float(p)
        result[f"{rank}_Recall"] = float(r)
        result[f"{rank}_F1"] = float(score)
        result[f"{rank}_Support"] = int(count)
    return result


def _fit_models(plan):
    estimators = {
        "Decision Tree": DecisionTreeRankClassifier(**MODEL_CONFIGURATIONS["Decision Tree"]),
        "Random Forest": RandomForestRankClassifier(**MODEL_CONFIGURATIONS["Random Forest"]),
        "Gradient Boosting Baseline": GradientBoostingRankClassifier(
            **{key: value for key, value in MODEL_CONFIGURATIONS["Gradient Boosting Baseline"].items() if key != "sample_weight"}
        ),
        "Gradient Boosting Balanced": GradientBoostingClassifier(
            **{key: value for key, value in MODEL_CONFIGURATIONS["Gradient Boosting Balanced"].items() if key != "sample_weight"}
        ),
    }
    outputs = {}
    for name, estimator in estimators.items():
        start = time.perf_counter()
        if name == "Gradient Boosting Balanced":
            weights = compute_sample_weight(class_weight="balanced", y=plan["y_train"])
            estimator.fit(plan["X_train"], plan["y_train"], sample_weight=weights)
        else:
            estimator.fit(plan["X_train"], plan["y_train"])
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        prediction = estimator.predict(plan["X_test"])
        predict_seconds = time.perf_counter() - start
        outputs[name] = {
            "predictions": np.asarray(prediction),
            "training_time_seconds": float(fit_seconds),
            "prediction_time_seconds": float(predict_seconds),
        }
    return outputs


def _json_write_exclusive(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def run_benchmark(transactions, *, output_dir=None):
    clean, plans, validation = _prepare_and_validate(transactions)
    if validation["status"] != "PASS" or not all(item["checks_passed"] for item in validation["cutoffs"].values()):
        raise RuntimeError("Pre-fit benchmark validation did not pass; no estimators were fitted.")

    if output_dir is None:
        run_name = f"customer_rank_90d_classifier_benchmark_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        output_path = EXPERIMENTS_ROOT / run_name
    else:
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path = output_path.resolve()
    try:
        output_path.relative_to(EXPERIMENTS_ROOT)
    except ValueError as exc:
        raise ValueError(f"Benchmark output must be under {EXPERIMENTS_ROOT}.") from exc
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(exist_ok=False)
    # Persist all pre-fit checks and per-cutoff IDs before any model fit begins.
    _json_write_exclusive(output_path / "pre_fit_validation.json", validation)

    result_rows = []
    prediction_rows = []
    confusion_matrices = {}
    for plan in plans:
        cutoff_key = plan["cutoff"].date().isoformat()
        confusion_matrices[cutoff_key] = {}
        outputs = _fit_models(plan)
        for model_name, output in outputs.items():
            prediction = output["predictions"]
            if not set(prediction).issubset(set(RANK_ORDER)) or len(prediction) != len(plan["test"]):
                raise RuntimeError(f"Invalid predictions for {model_name} at {cutoff_key}.")
            metrics = _metric_row(plan["y_test"], prediction)
            confusion_matrices[cutoff_key][model_name] = metrics.pop("Confusion_Matrix")
            result_rows.append(
                {
                    "Cutoff": cutoff_key,
                    "History_Start_Inclusive": plan["window_metadata"]["history_start"].isoformat(),
                    "History_End_Exclusive": plan["window_metadata"]["history_end_exclusive"].isoformat(),
                    "Future_Start_Inclusive": plan["window_metadata"]["future_start_inclusive"].isoformat(),
                    "Future_End_Exclusive": plan["window_metadata"]["future_end_exclusive"].isoformat(),
                    "Model": model_name,
                    "Evaluation_Set_ID": plan["evaluation_set_id"],
                    "Train_Customers": len(plan["train"]),
                    "Test_Customers": len(plan["test"]),
                    "q50": plan["thresholds"].q50,
                    "q85": plan["thresholds"].q85,
                    "Training_Time_Seconds": output["training_time_seconds"],
                    "Prediction_Time_Seconds": output["prediction_time_seconds"],
                    "Model_Parameters": json.dumps(MODEL_CONFIGURATIONS[model_name], sort_keys=True),
                    **metrics,
                }
            )
            for customer_id, actual, predicted in zip(plan["test"]["CustomerID"], plan["y_test"], prediction):
                prediction_rows.append(
                    {
                        "Cutoff": cutoff_key,
                        "Model": model_name,
                        "Evaluation_Set_ID": plan["evaluation_set_id"],
                        "CustomerID": _canonical_customer_id(customer_id),
                        "CustomerRank_true": actual,
                        "CustomerRank_predicted": predicted,
                    }
                )

    per_cutoff = pd.DataFrame(result_rows)
    prediction_frame = pd.DataFrame(prediction_rows)
    aggregate_rows = []
    aggregate_columns = [
        "Accuracy", "Macro_Precision", "Macro_Recall", "Macro_F1", "Balanced_Accuracy",
        *[f"{rank}_{metric}" for rank in RANK_ORDER for metric in ("Precision", "Recall", "F1")],
        "Training_Time_Seconds", "Prediction_Time_Seconds",
    ]
    for model_name, group in per_cutoff.groupby("Model", sort=False):
        row = {"Model": model_name, "Cutoff_Count": int(len(group))}
        for column in aggregate_columns:
            row[f"{column}_Mean"] = float(group[column].mean())
            row[f"{column}_Std"] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)

    metadata = {
        "experiment": "Customer-Ranking classifier benchmark with strict 90-day half-open windows",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_status_before_fit": "PASS",
        "pre_fit_validation_file": "pre_fit_validation.json",
        "feature_columns": list(FEATURE_COLUMNS),
        "history_window": "[cutoff - 90 days, cutoff)",
        "future_label_window": "[cutoff, cutoff + 90 days)",
        "cohort_rule": "customers with at least one valid transaction in the history window; future-window absence yields FutureSpend=0",
        "split_rule": "customer-level train/test split, test_size=0.2, random_state=42, independently at each cutoff",
        "threshold_rule": "positive train FutureSpend q50/q85; thresholds fixed and applied to both train and test labels",
        "model_configuration_tuning_used_test": False,
        "models_and_fixed_parameters": MODEL_CONFIGURATIONS,
        "source_data_valid_rows": int(len(clean)),
        "source_data_valid_min_timestamp": clean["InvoiceDate"].min().isoformat(),
        "source_data_valid_max_timestamp": clean["InvoiceDate"].max().isoformat(),
        "cutoffs": validation["cutoffs"],
        "sklearn_version": sklearn.__version__,
        "aggregate_standard_deviation_ddof": 1,
        "cross_cutoff_customer_overlap_is_possible": True,
        "cutoffs_are_independent_folds": False,
        "limitations": [
            "The three cutoff windows come from the same transaction source and can share customers.",
            "Cross-cutoff metrics are descriptive and must not be treated as independent-fold estimates.",
            "Measured runtimes depend on hardware and current system load.",
        ],
    }

    # Exclusive writes: this experiment path and every artifact must be new.
    per_cutoff.to_csv(output_path / "results_by_cutoff.csv", index=False, encoding="utf-8", mode="x")
    aggregate.to_csv(output_path / "aggregate_mean_std.csv", index=False, encoding="utf-8", mode="x")
    prediction_frame.to_csv(output_path / "predictions_by_cutoff.csv", index=False, encoding="utf-8", mode="x")
    _json_write_exclusive(output_path / "confusion_matrices.json", {"labels": list(RANK_ORDER), "per_cutoff": confusion_matrices})
    _json_write_exclusive(output_path / "metadata.json", metadata)
    return {"per_cutoff": per_cutoff, "aggregate": aggregate, "metadata": metadata, "output_dir": output_path}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the fixed-window Customer-Ranking classifier benchmark.")
    parser.add_argument("--data", default=None, help="Transaction Excel path; defaults to data/raw/Online Retail.xlsx.")
    parser.add_argument("--output-dir", default=None, help="New output directory under outputs/experiments/.")
    args = parser.parse_args(argv)
    result = run_benchmark(load_data(args.data), output_dir=args.output_dir)
    print("Results by cutoff:")
    print(result["per_cutoff"].to_string(index=False))
    print("Aggregate mean/std:")
    print(result["aggregate"].to_string(index=False))
    print(f"Benchmark artifacts saved to {result['output_dir']}")


if __name__ == "__main__":
    main()
