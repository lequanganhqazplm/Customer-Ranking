"""Synthetic-only tests for the Decision Tree CustomerRank pipeline."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier
from streamlit.testing.v1 import AppTest

from src.evaluate import COMMON_METRICS
from src.feature_engineering import FEATURE_COLUMNS, RANK_ORDER, split_customer_dataset
from src.models.decision_tree_model import DecisionTreeRankClassifier
from src.predict import build_model_input
from src.train import _evaluation_set_id as kmeans_evaluation_set_id
from src.train_decision_tree import run_training, save_experiment


def synthetic_transactions(customer_count=64):
    rows = []
    base_date = pd.Timestamp("2020-01-01")
    group_spends = (20.0, 100.0, 600.0, 3000.0)
    for customer_number in range(customer_count):
        customer_id = 20000 + customer_number
        group = customer_number % 4
        history_day = 10 + group * 15
        for purchase in range(1 + group):
            rows.append(
                {
                    "InvoiceNo": f"H-{customer_id}-{purchase}",
                    "StockCode": f"SKU-{group}-{purchase}",
                    "InvoiceDate": base_date + pd.Timedelta(days=history_day + purchase),
                    "Quantity": 1 + purchase,
                    "UnitPrice": 5.0 + group * 8 + customer_number % 7,
                    "CustomerID": customer_id,
                }
            )
        if customer_number % 6 != 0:
            amount = group_spends[group] + (customer_number % 8) * (2 + group * 3)
            rows.append(
                {
                    "InvoiceNo": f"F-{customer_id}",
                    "StockCode": f"FUTURE-{group}",
                    "InvoiceDate": base_date + pd.Timedelta(days=150),
                    "Quantity": 1,
                    "UnitPrice": amount,
                    "CustomerID": customer_id,
                }
            )

    rows.append(
        {
            "InvoiceNo": "OBSERVATION-END",
            "StockCode": "END",
            "InvoiceDate": base_date + pd.Timedelta(days=180),
            "Quantity": 1,
            "UnitPrice": 1.0,
            "CustomerID": 99999,
        }
    )
    return pd.DataFrame(rows)


@pytest.fixture
def trained_result():
    return run_training(synthetic_transactions(), random_state=42)


def test_training_split_thresholds_and_evaluation_set_follow_shared_contract(trained_result):
    dataset = trained_result["dataset"]
    train, test, thresholds = split_customer_dataset(dataset, test_size=0.2, random_state=42)

    assert not set(train["CustomerID"]) & set(test["CustomerID"])
    assert set(trained_result["train_customer_ids"]) == set(train["CustomerID"])
    assert set(trained_result["test_customer_ids"]) == set(test["CustomerID"])
    assert trained_result["thresholds"] == thresholds
    assert trained_result["metadata"]["rank_thresholds"] == thresholds.as_dict()

    normalized_ids = sorted(str(value) for value in test["CustomerID"])
    expected_id = hashlib.sha256("\n".join(normalized_ids).encode("utf-8")).hexdigest()[:16]
    assert expected_id == kmeans_evaluation_set_id(test["CustomerID"])
    assert trained_result["result_row"]["Evaluation_Set_ID"] == expected_id
    assert trained_result["metadata"]["evaluation_set_id"] == expected_id


def test_classifier_uses_only_six_historical_features_and_valid_rank_labels(trained_result):
    dataset = trained_result["dataset"]
    model = trained_result["model"]

    assert tuple(model.feature_names_in_) == FEATURE_COLUMNS
    assert isinstance(model.estimator_, DecisionTreeClassifier)
    assert not hasattr(model, "scaler_")
    assert set(trained_result["predictions"]["CustomerRank_predicted"]).issubset(RANK_ORDER)
    assert tuple(build_model_input(dataset).columns) == FEATURE_COLUMNS
    with pytest.raises(ValueError, match="exactly these ordered features"):
        model.fit(dataset, dataset["CustomerRank"])


def test_future_target_changes_do_not_change_model_inputs_or_train_thresholds(trained_result):
    dataset = trained_result["dataset"]
    original_features = build_model_input(dataset)
    changed = dataset.copy()
    test_mask = changed["CustomerID"].isin(trained_result["test_customer_ids"])
    changed.loc[test_mask, "FutureSpend"] = changed.loc[test_mask, "FutureSpend"] * 1000 + 100
    changed["ReturnedNext90Days"] = 1 - changed["ReturnedNext90Days"]
    changed["CustomerRank"] = "Diamond"

    pd.testing.assert_frame_equal(build_model_input(changed), original_features)
    _, _, changed_thresholds = split_customer_dataset(changed, test_size=0.2, random_state=42)
    assert changed_thresholds == trained_result["thresholds"]


def test_training_emits_all_five_common_metrics(trained_result):
    assert set(COMMON_METRICS).issubset(trained_result["metrics"])
    assert all(np.isfinite(trained_result["metrics"][metric]) for metric in COMMON_METRICS)
    assert trained_result["result_row"]["Status"] == "completed"
    assert len(trained_result["predictions"]) == trained_result["metadata"]["n_test_customers"]


def test_save_experiment_writes_only_into_new_run_directory(trained_result, tmp_path):
    destination = tmp_path / "new-run"
    saved = save_experiment(trained_result, destination)

    assert saved == destination
    assert {path.name for path in destination.iterdir()} == {
        "model.joblib",
        "ranking_metrics.csv",
        "test_predictions.csv",
        "run_metadata.json",
    }
    metadata = json.loads((destination / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["evaluation_set_id"] == trained_result["result_row"]["Evaluation_Set_ID"]


def test_save_experiment_refuses_existing_run_without_changing_it(trained_result, tmp_path):
    destination = tmp_path / "existing-run"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("protected\n", encoding="utf-8")
    before = sentinel.read_bytes()

    with pytest.raises(FileExistsError, match="existing run directory"):
        save_experiment(trained_result, destination)

    assert sentinel.read_bytes() == before
    assert list(destination.iterdir()) == [sentinel]


def test_model_rejects_missing_or_invalid_rank_labels():
    features = pd.DataFrame({column: [float(i), float(i + 1)] for i, column in enumerate(FEATURE_COLUMNS)})
    model = DecisionTreeRankClassifier()
    with pytest.raises(ValueError, match="outside the four customer ranks"):
        model.fit(features, ["Bronze", "VIP"])
    with pytest.raises(ValueError, match="non-missing"):
        model.fit(features, ["Bronze", None])


def test_streamlit_button_trains_decision_tree_and_saves_a_new_run(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    raw_data = data_root / "raw" / "Online Retail.xlsx"
    raw_data.parent.mkdir(parents=True)
    raw_data.write_bytes(b"test placeholder")
    outputs_root = tmp_path / "outputs"
    monkeypatch.setenv("CUSTOMER_RANKING_DATA_ROOT", str(data_root))
    monkeypatch.setenv("CUSTOMER_RANKING_OUTPUTS_ROOT", str(outputs_root))
    monkeypatch.setattr("src.data_loader.load_data", lambda path=None: synthetic_transactions())
    monkeypatch.setattr("streamlit.rerun", lambda: None)

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path).run()
    assert not app.exception

    train_selector = next(widget for widget in app.selectbox if widget.label == "Chọn mô hình huấn luyện")
    train_selector.set_value("Cây quyết định (Decision Tree)")
    button = next(widget for widget in app.button if widget.label == "Bắt đầu huấn luyện")
    app = button.click().run()

    assert not app.exception
    run_dirs = list((outputs_root / "runs").glob("rank-decision_tree-*"))
    assert len(run_dirs) == 1
    assert {path.name for path in run_dirs[0].iterdir()} == {
        "model.joblib",
        "ranking_metrics.csv",
        "test_predictions.csv",
        "run_metadata.json",
    }
    metadata = json.loads((run_dirs[0] / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["model"] == "Decision Tree"
    assert metadata["feature_importances"]

    navigation = next(widget for widget in app.radio if widget.label == "Điều hướng")
    navigation.set_value("Mô hình").run()
    model_selector = next(widget for widget in app.selectbox if widget.label == "Chọn mô hình")
    model_selector.set_value("Cây quyết định (Decision Tree)").run()
    assert not app.exception
    assert any(metric.label == "Độ chính xác" for metric in app.metric)
    assert any("Mức độ đóng góp của các đặc trưng" in item.value for item in app.subheader)
    assert any("Phân bố hạng thực tế và dự đoán" in item.value for item in app.subheader)
