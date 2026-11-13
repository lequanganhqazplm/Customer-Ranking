"""Synthetic-only integration tests for Gradient Boosting training and artifacts."""

import json

import numpy as np
import pandas as pd
import pytest
from src.dashboard_helpers import load_gb_run, validate_gb_predictions
from src.evaluate import COMMON_METRICS
from src.feature_engineering import FEATURE_COLUMNS, split_customer_dataset
from src.train_gradient_boosting import run_training, save_experiment


def synthetic_transactions(customer_count=80):
    rows = []
    base_date = pd.Timestamp("2020-01-01")
    group_spends = (20.0, 100.0, 600.0, 3000.0)
    for number in range(customer_count):
        customer_id = 30000 + number
        group = number % 4
        for purchase in range(1 + group):
            rows.append({
                "InvoiceNo": f"H-{customer_id}-{purchase}",
                "StockCode": f"SKU-{group}-{purchase}",
                "InvoiceDate": base_date + pd.Timedelta(days=10 + group * 15 + purchase),
                "Quantity": 1 + purchase,
                "UnitPrice": 5.0 + group * 8 + number % 7,
                "CustomerID": customer_id,
            })
        if number % 6 != 0:
            spend = group_spends[group] + (number % 8) * (2 + group * 3)
            rows.append({
                "InvoiceNo": f"F-{customer_id}", "StockCode": f"FUTURE-{group}",
                "InvoiceDate": base_date + pd.Timedelta(days=150), "Quantity": 1,
                "UnitPrice": spend, "CustomerID": customer_id,
            })
    rows.append({
        "InvoiceNo": "OBSERVATION-END", "StockCode": "END",
        "InvoiceDate": base_date + pd.Timedelta(days=180), "Quantity": 1,
        "UnitPrice": 1.0, "CustomerID": 99999,
    })
    return pd.DataFrame(rows)


@pytest.fixture
def trained_result():
    return run_training(
        synthetic_transactions(), random_state=42,
        n_estimators=10, min_samples_split=8, min_samples_leaf=2,
    )


def test_pipeline_uses_customer_split_and_complete_classification_metrics(trained_result):
    train, test, thresholds = split_customer_dataset(trained_result["dataset"], random_state=42)
    assert not set(train.CustomerID) & set(test.CustomerID)
    assert set(trained_result["train_customer_ids"]) == set(train.CustomerID)
    assert set(trained_result["test_customer_ids"]) == set(test.CustomerID)
    assert trained_result["thresholds"] == thresholds
    assert set(COMMON_METRICS).issubset(trained_result["metrics"])
    assert all(np.isfinite(trained_result["metrics"][metric]) for metric in COMMON_METRICS)
    assert trained_result["metadata"]["evaluation_set_id"] == trained_result["result_row"]["Evaluation_Set_ID"]


def test_pipeline_records_model_metadata_predictions_and_feature_importance(trained_result):
    metadata = trained_result["metadata"]
    assert metadata["model"] == "Gradient Boosting"
    assert metadata["feature_columns"] == list(FEATURE_COLUMNS)
    assert metadata["n_train_customers"] + metadata["n_test_customers"] == len(trained_result["dataset"])
    assert metadata["model_parameters"] == {
        "n_estimators": 10, "learning_rate": 0.1, "max_depth": 3,
        "min_samples_split": 8, "min_samples_leaf": 2, "random_state": 42,
    }
    assert len(metadata["feature_importances"]) == len(FEATURE_COLUMNS)
    assert metadata["training_time_seconds"] >= 0
    assert metadata["prediction_time_seconds"] >= 0
    assert set(trained_result["predictions"]["CustomerRank_predicted"]).issubset(
        {"Diamond", "Gold", "Silver", "Bronze"}
    )


def test_save_refuses_existing_run_and_dashboard_loads_saved_artifacts(trained_result, tmp_path):
    destination = tmp_path / "rank-gradient-boosting-run-01"
    save_experiment(trained_result, destination)
    assert {item.name for item in destination.iterdir()} == {
        "model.joblib", "ranking_metrics.csv", "test_predictions.csv", "run_metadata.json",
    }
    metadata, predictions, loaded_path, error = load_gb_run(tmp_path)
    assert error is None
    assert loaded_path == destination
    assert metadata["model"] == "Gradient Boosting"
    assert len(predictions) == metadata["n_test_customers"]
    validated, error = validate_gb_predictions(trained_result["predictions"])
    assert error is None
    assert len(validated) == len(trained_result["predictions"])

    sentinel = destination / "run_metadata.json"
    before = sentinel.read_bytes()
    with pytest.raises(FileExistsError):
        save_experiment(trained_result, destination)
    assert sentinel.read_bytes() == before


def test_dashboard_loader_reports_invalid_or_incomplete_run(tmp_path):
    run_path = tmp_path / "incomplete"
    run_path.mkdir()
    (run_path / "ranking_metrics.csv").write_text("Model\nGradient Boosting\n", encoding="utf-8")
    (run_path / "run_metadata.json").write_text(
        json.dumps({"model": "Gradient Boosting", "n_test_customers": 1, "evaluation_set_id": "abc", "feature_columns": list(FEATURE_COLUMNS)}),
        encoding="utf-8",
    )
    metadata, predictions, loaded_path, error = load_gb_run(tmp_path)
    assert metadata["model"] == "Gradient Boosting"
    assert predictions is None and loaded_path == run_path
    assert "dự đoán" in error


def test_shared_training_dispatch_accepts_gradient_boosting_alias(monkeypatch):
    import src.train as shared_train
    import src.train_gradient_boosting as gradient_boosting

    observed = {}

    def fake_train(dataset, *, test_size, random_state):
        observed["dataset_size"] = len(dataset)
        return {"metadata": {"model": "Gradient Boosting"}}

    monkeypatch.setattr(gradient_boosting, "train_gradient_boosting_rank_model", fake_train)
    result = shared_train.run_training(
        synthetic_transactions(), model_name="gb", test_size=0.25, random_state=7,
    )

    assert observed["dataset_size"] == len(result["dataset"])
    assert result["metadata"]["model"] == "Gradient Boosting"
