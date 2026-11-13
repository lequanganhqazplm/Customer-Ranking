"""Synthetic-only tests for Random Forest pipeline, dispatch, and artifacts."""

import json

import numpy as np
import pandas as pd
import pytest

from src.dashboard_helpers import load_rf_run, validate_rf_predictions
from src.evaluate import COMMON_METRICS
from src.feature_engineering import FEATURE_COLUMNS, split_customer_dataset
from src.train_random_forest import run_training, save_experiment


def synthetic_transactions(customer_count=80):
    rows = []
    base_date = pd.Timestamp("2020-01-01")
    spends = (20.0, 100.0, 600.0, 3000.0)
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
            rows.append({
                "InvoiceNo": f"F-{customer_id}", "StockCode": f"FUTURE-{group}",
                "InvoiceDate": base_date + pd.Timedelta(days=150), "Quantity": 1,
                "UnitPrice": spends[group] + (number % 8) * (2 + group * 3),
                "CustomerID": customer_id,
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
        synthetic_transactions(), random_state=42, n_estimators=12, n_jobs=1,
        min_samples_split=8, min_samples_leaf=2,
    )


def test_pipeline_splits_by_customer_and_records_train_only_thresholds(trained_result):
    train, test, thresholds = split_customer_dataset(trained_result["dataset"], random_state=42)
    assert not set(train.CustomerID) & set(test.CustomerID)
    assert set(trained_result["train_customer_ids"]) == set(train.CustomerID)
    assert set(trained_result["test_customer_ids"]) == set(test.CustomerID)
    assert trained_result["thresholds"] == thresholds
    assert set(COMMON_METRICS).issubset(trained_result["metrics"])
    assert all(np.isfinite(trained_result["metrics"][metric]) for metric in COMMON_METRICS)


def test_pipeline_records_features_parameters_predictions_and_importances(trained_result):
    metadata = trained_result["metadata"]
    assert metadata["model"] == "Random Forest"
    assert metadata["feature_columns"] == list(FEATURE_COLUMNS)
    assert metadata["model_parameters"]["n_estimators"] == 12
    assert len(metadata["feature_importances"]) == len(FEATURE_COLUMNS)
    assert len(trained_result["predictions"]) == metadata["n_test_customers"]
    validated, error = validate_rf_predictions(trained_result["predictions"])
    assert error is None
    assert len(validated) == len(trained_result["predictions"])


def test_save_creates_new_artifacts_and_dashboard_loader_reads_them(trained_result, tmp_path):
    destination = tmp_path / "random-forest-run-01"
    save_experiment(trained_result, destination)
    assert {item.name for item in destination.iterdir()} == {
        "model.joblib", "ranking_metrics.csv", "test_predictions.csv", "run_metadata.json",
    }
    metadata, predictions, loaded_path, error = load_rf_run(tmp_path)
    assert error is None
    assert loaded_path == destination
    assert metadata["model"] == "Random Forest"
    assert len(predictions) == metadata["n_test_customers"]
    before = (destination / "run_metadata.json").read_bytes()
    with pytest.raises(FileExistsError):
        save_experiment(trained_result, destination)
    assert (destination / "run_metadata.json").read_bytes() == before


def test_dashboard_loader_rejects_incomplete_random_forest_run(tmp_path):
    run_path = tmp_path / "incomplete"
    run_path.mkdir()
    (run_path / "ranking_metrics.csv").write_text("Model\nRandom Forest\n", encoding="utf-8")
    (run_path / "run_metadata.json").write_text(
        json.dumps({"model": "Random Forest", "model_type": "Classification"}), encoding="utf-8"
    )
    metadata, predictions, loaded_path, error = load_rf_run(tmp_path)
    assert metadata["model"] == "Random Forest"
    assert predictions is None and loaded_path == run_path
    assert "train" in error


def test_shared_training_dispatch_accepts_random_forest_alias(monkeypatch):
    import src.train as shared_train
    import src.train_random_forest as random_forest

    observed = {}

    def fake_train(dataset, *, test_size, random_state):
        observed["dataset_size"] = len(dataset)
        return {"metadata": {"model": "Random Forest"}}

    monkeypatch.setattr(random_forest, "train_random_forest_rank_model", fake_train)
    result = shared_train.run_training(
        synthetic_transactions(), model_name="rf", test_size=0.25, random_state=7,
    )
    assert observed["dataset_size"] == len(result["dataset"])
    assert result["metadata"]["model"] == "Random Forest"
