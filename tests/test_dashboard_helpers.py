"""Tests for report validation and dynamic dashboard cluster labels."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the staged helper module importable when the test is run from this folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard_helpers import (
    MODEL_CATALOG,
    RANK_DISPLAY_NAMES,
    build_cluster_labels,
    comparable_model_results,
    read_csv_report,
    load_dt_run,
    reconcile_cluster_summary,
    validate_dt_predictions,
    validate_test_predictions,
    validate_model_results,
    validate_customer_report,
    validate_evaluation_report,
    validate_feature_report,
)


@pytest.fixture
def customer_rows():
    return pd.DataFrame(
        {
            "CustomerID": [1001, 1002, 1003],
            "Cluster": [1, 1, 0],
            "Recency": [10, 20, 90],
            "Frequency": [8, 4, 1],
            "Monetary": [500.0, 200.0, 50.0],
            "TotalQuantity": [40, 15, 2],
            "UniqueProducts": [12, 8, 1],
            "AvgOrderValue": [62.5, 50.0, 50.0],
        }
    )


def test_read_csv_report_handles_missing_empty_and_missing_columns(tmp_path):
    missing, error = read_csv_report(tmp_path / "missing.csv", ("Cluster",))
    assert missing is None
    assert "Không tìm thấy" in error

    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("", encoding="utf-8")
    empty, error = read_csv_report(empty_path)
    assert empty is None
    assert "không có dữ liệu" in error

    wrong_path = tmp_path / "wrong.csv"
    wrong_path.write_text("CustomerID\n1001\n", encoding="utf-8")
    wrong, error = read_csv_report(wrong_path, ("Cluster",))
    assert wrong is None
    assert "Cluster" in error


def test_read_csv_report_handles_malformed_csv(tmp_path):
    malformed_path = tmp_path / "malformed.csv"
    malformed_path.write_text('a,b\n"unterminated,2\n', encoding="utf-8")
    malformed, error = read_csv_report(malformed_path)
    assert malformed is None
    assert "Không thể đọc" in error


def test_validate_customer_report_normalizes_ids_and_rejects_malformed_rows(customer_rows):
    valid, error = validate_customer_report(customer_rows)
    assert error is None
    assert valid["CustomerID"].tolist() == ["1001", "1002", "1003"]
    assert set(valid["Cluster"]) == {0, 1}

    missing_column, error = validate_customer_report(customer_rows.drop(columns="Monetary"))
    assert missing_column is None
    assert "Monetary" in error

    bad_value = customer_rows.copy()
    bad_value["Cluster"] = bad_value["Cluster"].astype(object)
    bad_value.loc[0, "Cluster"] = "not-a-cluster"
    validated, error = validate_customer_report(bad_value)
    assert validated is None
    assert "Cluster" in error


def test_validate_feature_report_does_not_require_or_infer_a_cluster(customer_rows):
    features = customer_rows.drop(columns="Cluster")
    features["CustomerID"] = features["CustomerID"].astype(float)

    validated, error = validate_feature_report(features)

    assert error is None
    assert validated["CustomerID"].tolist() == ["1001", "1002", "1003"]
    assert "Cluster" not in validated.columns
    invalid = features.assign(Monetary="not-a-number")
    validated, error = validate_feature_report(invalid)
    assert validated is None
    assert "Monetary" in error


def test_cluster_labels_follow_summary_statistics_and_preserve_numeric_ids(customer_rows):
    customers, error = validate_customer_report(customer_rows)
    assert error is None
    summary, _ = reconcile_cluster_summary(None, customers)

    labels = build_cluster_labels(summary)

    assert labels[1].startswith("Chi tiêu trung vị cao nhất")
    assert labels[0].startswith("Chi tiêu trung vị thấp nhất")
    assert "VIP" not in " ".join(labels.values())
    assert customers["Cluster"].tolist() == [1, 1, 0]


def test_reconcile_cluster_summary_fills_missing_cluster_rows(customer_rows):
    customers, error = validate_customer_report(customer_rows)
    assert error is None
    summary, _ = reconcile_cluster_summary(None, customers)
    incomplete = summary[summary["Cluster"] == 1]

    reconciled, warning = reconcile_cluster_summary(incomplete, customers)

    assert reconciled["Cluster"].tolist() == [0, 1]
    assert reconciled["n_customers"].tolist() == [1, 2]
    assert "bị thiếu" in warning


def test_reconcile_cluster_summary_falls_back_on_malformed_report(customer_rows):
    customers, error = validate_customer_report(customer_rows)
    assert error is None
    bad_summary = pd.DataFrame({"Cluster": [0], "Monetary_mean": ["bad"]})

    summary, warning = reconcile_cluster_summary(bad_summary, customers)

    assert summary["Cluster"].tolist() == [0, 1]
    assert "thiếu dữ liệu" in warning


def test_evaluation_report_validates_required_numeric_columns():
    valid, error = validate_evaluation_report(
        pd.DataFrame({"k": [2], "inertia": [10.0], "silhouette_score": [0.5]})
    )
    assert error is None
    assert valid.iloc[0].to_dict() == {"k": 2.0, "inertia": 10.0, "silhouette_score": 0.5}

    invalid, error = validate_evaluation_report(pd.DataFrame({"k": [2]}))
    assert invalid is None
    assert "inertia" in error


def test_model_catalog_contains_exactly_ten_distinct_algorithms():
    names = [entry["name"] for entry in MODEL_CATALOG]
    assert len(names) == 10
    assert len(set(names)) == 10
    assert names[0] == "K-Means"
    assert sum(entry["implemented"] for entry in MODEL_CATALOG) == 2
    assert next(item for item in MODEL_CATALOG if item["name"] == "Decision Tree")["implemented"]


def test_model_comparison_rejects_legacy_or_incomplete_schema_and_warns_on_test_mismatch():
    legacy, error = validate_model_results(
        pd.DataFrame({"Model": ["K-Means"], "Silhouette": [0.8], "Inertia": [10]})
    )
    assert legacy is None
    assert "ModelType" in error

    rows = pd.DataFrame(
        {
            "Model": ["K-Means", "Logistic Regression"],
            "ModelType": ["Clustering", "Classification"],
            "Accuracy": [0.7, 0.6],
            "Precision_Macro": [0.6, 0.5],
            "Recall_Macro": [0.7, 0.6],
            "F1_Macro": [0.6, 0.5],
            "Balanced_Accuracy": [0.7, 0.6],
            "Training_Time": [1.0, 2.0],
            "Prediction_Time": [0.1, 0.2],
            "Status": ["completed", "completed"],
            "Evaluation_Set_ID": ["test-a", "test-b"],
        }
    )
    result, warning = validate_model_results(rows)
    assert result is not None
    assert "nhiều tập test" in warning


def test_comparison_includes_only_completed_results_from_one_same_test_set():
    rows = pd.DataFrame(
        {
            "Model": ["K-Means", "SVC", "Ridge Regression"],
            "ModelType": ["Clustering", "Classification", "Regression"],
            "Accuracy": [0.7, 0.6, None],
            "Precision_Macro": [0.6, 0.5, None],
            "Recall_Macro": [0.7, 0.6, None],
            "F1_Macro": [0.6, 0.5, None],
            "Balanced_Accuracy": [0.7, 0.6, None],
            "Training_Time": [1.0, 2.0, None],
            "Prediction_Time": [0.1, 0.2, None],
            "Status": ["completed", "completed", "not_implemented"],
            "Evaluation_Set_ID": ["test-a", "test-b", None],
        }
    )
    same_set, excluded, warning = comparable_model_results(rows)
    assert same_set["Model"].tolist() == ["K-Means"]
    assert excluded == ["SVC"]
    assert "SVC" in warning


def test_test_predictions_validate_four_labels_and_preserve_cluster_ids():
    data = pd.DataFrame(
        {
            "CustomerID": [14646, 14911],
            "CustomerRank_true": ["Diamond", "Gold"],
            "CustomerRank_predicted": ["Gold", "Silver"],
            "Cluster": [3, 1],
        }
    )
    result, error = validate_test_predictions(data)
    assert error is None
    assert result["CustomerID"].tolist() == ["14646", "14911"]
    assert result["ClusterID"].tolist() == [3, 1]
    assert result["CustomerRank_predicted"].tolist() == ["Gold", "Silver"]
    assert set(RANK_DISPLAY_NAMES) == {"Diamond", "Gold", "Silver", "Bronze"}

    malformed, error = validate_test_predictions(data.assign(CustomerRank_predicted="VIP"))
    assert malformed is None
    assert "nhãn không hợp lệ" in error


def test_decision_tree_predictions_validate_without_a_cluster_column():
    data = pd.DataFrame(
        {
            "CustomerID": [14646, 14911],
            "CustomerRank_true": ["Diamond", "Gold"],
            "CustomerRank_predicted": ["Gold", "Silver"],
        }
    )
    result, error = validate_dt_predictions(data)
    assert error is None
    assert result["CustomerID"].tolist() == ["14646", "14911"]
    assert "ClusterID" not in result.columns

    invalid, error = validate_dt_predictions(data.assign(CustomerRank_predicted="VIP"))
    assert invalid is None
    assert "nhãn không hợp lệ" in error


def test_load_dt_run_reads_and_validates_latest_decision_tree_artifacts(tmp_path):
    run_path = tmp_path / "decision-tree-run"
    run_path.mkdir()
    (run_path / "ranking_metrics.csv").write_text("Model\nDecision Tree\n", encoding="utf-8")
    (run_path / "run_metadata.json").write_text(
        '{"model": "Decision Tree", "n_test_customers": 2}', encoding="utf-8"
    )
    (run_path / "test_predictions.csv").write_text(
        "CustomerID,CustomerRank_true,CustomerRank_predicted\n"
        "14646,Diamond,Gold\n14911,Gold,Gold\n",
        encoding="utf-8",
    )

    metadata, predictions, loaded_path, error = load_dt_run(tmp_path)

    assert error is None
    assert metadata["model"] == "Decision Tree"
    assert predictions["CustomerID"].tolist() == ["14646", "14911"]
    assert loaded_path == run_path


def test_comparison_rejects_unidentified_test_set():
    rows = pd.DataFrame(
        {
            "Model": ["K-Means"], "ModelType": ["Clustering"],
            "Accuracy": [0.7], "Precision_Macro": [0.6], "Recall_Macro": [0.7],
            "F1_Macro": [0.6], "Balanced_Accuracy": [0.7], "Training_Time": [1.0],
            "Prediction_Time": [0.1], "Status": ["completed"], "Evaluation_Set_ID": [None],
        }
    )
    result, excluded, warning = comparable_model_results(rows)
    assert result.empty
    assert excluded == ["K-Means"]
    assert "Evaluation_Set_ID" in warning
