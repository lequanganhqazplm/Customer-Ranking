"""Synthetic-only tests for leakage boundaries and K-Means rank mapping."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import FEATURE_COLUMNS, split_customer_dataset
from src.models.kmeans_model import ClusterRankMapper, KMeansRankClassifier
from src.predict import build_model_input, predict_customer_rank
from src.train import run_training, save_experiment
from src import train_kmeans as legacy_kmeans


def synthetic_transactions(customer_count=60):
    rows = []
    for customer_number in range(customer_count):
        customer_id = 20000 + customer_number
        group = customer_number % 4
        base_amount = [12.0, 35.0, 90.0, 240.0][group] + (customer_number // 4)
        history_days = [15, 35, 65, 82][group]
        for purchase in range(1 + group):
            rows.append(
                {
                    "InvoiceNo": f"H-{customer_id}-{purchase}",
                    "StockCode": f"SKU-{group}-{purchase}",
                    "InvoiceDate": pd.Timestamp("2020-01-01") + pd.Timedelta(days=history_days - purchase * 3),
                    "Quantity": 1 + purchase,
                    "UnitPrice": base_amount,
                    "CustomerID": customer_id,
                }
            )
        if customer_number % 6 != 0:
            future_amount = 0.0 if customer_number % 5 == 0 else 40.0 + group * 180 + customer_number
            rows.append(
                {
                    "InvoiceNo": f"F-{customer_id}",
                    "StockCode": f"NEXT-{group}",
                    "InvoiceDate": pd.Timestamp("2020-01-01") + pd.Timedelta(days=150),
                    "Quantity": 1,
                    "UnitPrice": future_amount,
                    "CustomerID": customer_id,
                }
            )
    rows.append(
        {
            "InvoiceNo": "END",
            "StockCode": "END",
            "InvoiceDate": pd.Timestamp("2020-06-29"),
            "Quantity": 1,
            "UnitPrice": 1.0,
            "CustomerID": 99999,
        }
    )
    return pd.DataFrame(rows)


def test_training_uses_train_only_thresholds_scaler_and_feature_based_cluster_map():
    result = run_training(synthetic_transactions(), n_clusters=4, random_state=42)
    dataset = result["dataset"]
    train, test, thresholds = split_customer_dataset(dataset, random_state=42)

    assert not set(train["CustomerID"]) & set(test["CustomerID"])
    assert result["metadata"]["rank_thresholds"] == thresholds.as_dict()
    np.testing.assert_allclose(result["model"].scaler_.mean_, train[list(FEATURE_COLUMNS)].mean().to_numpy())
    mapping = result["model"].rank_mapper_
    assert mapping.mapping_strategy_ == "train_rfm_percentile_composite"
    assert set(mapping.cluster_to_rank_.values()) == {"Diamond", "Gold", "Silver", "Bronze"}
    assert set(mapping.cluster_profiles_["ClusterID"]) == set(mapping.cluster_to_rank_)
    assert result["metadata"]["cluster_mapping_formula"] == (
        "mean(percentile(Monetary), percentile(Frequency), 1 - percentile(Recency))"
    )
    assert result["metadata"]["n_test_customers"] == len(test)
    assert result["result_row"]["Status"] == "completed"
    assert result["result_row"]["Evaluation_Set_ID"]
    assert set(result["metadata"]["train_rank_distribution"]) == {"Diamond", "Gold", "Silver", "Bronze"}
    assert set(result["metadata"]["test_rank_distribution"]) == {"Diamond", "Gold", "Silver", "Bronze"}
    assert sum(item["count"] for item in result["metadata"]["test_rank_distribution"].values()) == len(test)
    # This synthetic split can lack a true test tier; the pipeline records the
    # zero count and must not manufacture examples to fill it.
    assert set(result["predictions"]["CustomerRank_predicted"]).issubset(
        {"Diamond", "Gold", "Silver", "Bronze"}
    )
    assert set(result["metadata"]["train_rank_distribution"]) == {"Diamond", "Gold", "Silver", "Bronze"}
    assert set(result["metadata"]["test_rank_distribution"]) == {"Diamond", "Gold", "Silver", "Bronze"}


def test_dataset_features_exclude_future_targets_and_identifiers():
    result = run_training(synthetic_transactions(), n_clusters=4)
    features = build_model_input(result["dataset"])
    assert tuple(features.columns) == FEATURE_COLUMNS
    assert not {"CustomerID", "FutureSpend", "ReturnedNext90Days", "CustomerRank"}.intersection(features.columns)


def test_kmeans_rejects_leaking_columns_and_unknown_cluster_is_explicit():
    X = pd.DataFrame({"Recency": [1, 2, 3, 4], "FutureSpend": [0, 10, 20, 30]})
    model = KMeansRankClassifier(n_clusters=4)
    with pytest.raises(ValueError, match="cannot be model inputs"):
        model.fit(X)

    training_features = pd.DataFrame(
        {
            "Monetary": [10, 20, 30, 40, 50, 60, 70, 80],
            "Frequency": [1, 1, 2, 2, 3, 3, 4, 4],
            "Recency": [100, 90, 70, 60, 40, 30, 20, 10],
        }
    )
    mapper = ClusterRankMapper().fit([2, 2, 5, 5, 8, 8, 12, 12], training_features)
    assert mapper.cluster_to_rank_ == {2: "Bronze", 5: "Silver", 8: "Gold", 12: "Diamond"}
    ranks, unknown = mapper.map_with_status([2, 999])
    assert ranks[0] == "Bronze"
    assert ranks[1] is None
    assert unknown == [999]
    with pytest.raises(ValueError, match="No train-derived business mapping"):
        mapper.predict([999])


def test_prediction_adapter_supports_classification_regression_and_clustering():
    class Classifier:
        def predict(self, X):
            return np.asarray(["Silver"] * len(X))

    class Regressor:
        def predict(self, X):
            return np.asarray([-1.0, 300.0, 1500.0])[: len(X)]

    class ClusterRanker:
        def predict(self, X):
            return np.asarray(["Bronze"] * len(X))

    X = np.zeros((3, 1))
    assert predict_customer_rank(Classifier(), X, "classification").tolist() == ["Silver"] * 3
    assert predict_customer_rank(Regressor(), X, "regression", (500, 1500)).tolist() == ["Bronze", "Silver", "Diamond"]
    assert predict_customer_rank(ClusterRanker(), X, "clustering").tolist() == ["Bronze"] * 3
    with pytest.raises(ValueError, match="training-derived"):
        predict_customer_rank(Regressor(), X, "regression")


def test_saving_run_refuses_existing_directory(tmp_path):
    destination = tmp_path / "run"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="existing run directory"):
        save_experiment({}, destination)


def test_saving_run_writes_only_to_a_new_explicit_directory(tmp_path):
    result = run_training(synthetic_transactions(), n_clusters=4)
    destination = save_experiment(result, tmp_path / "new-run")

    assert (destination / "model.joblib").is_file()
    assert (destination / "ranking_metrics.csv").is_file()
    assert (destination / "test_predictions.csv").is_file()
    assert (destination / "run_metadata.json").is_file()


def test_legacy_kmeans_preflight_stops_before_replacing_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(legacy_kmeans, "PROJECT_ROOT", tmp_path)
    protected = tmp_path / "outputs" / "metrics" / "model_metrics.csv"
    protected.parent.mkdir(parents=True)
    protected.write_text("keep\n", encoding="utf-8")
    before = protected.read_bytes()

    with pytest.raises(FileExistsError, match="stopped before reading/training"):
        legacy_kmeans._ensure_output_paths_available()

    assert protected.read_bytes() == before
