"""Unit tests for the shared four-class rank evaluation API."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    COMMON_METRICS,
    evaluate_classification,
    evaluate_clustering,
    evaluate_regression,
    make_model_result,
    write_model_results,
)
from src.feature_engineering import RankThresholds


def test_classification_returns_five_metrics_and_records_absent_classes():
    result = evaluate_classification(
        ["Bronze", "Silver", "Gold", "Diamond"],
        ["Bronze", "Bronze", "Gold", "Bronze"],
    )
    assert set(COMMON_METRICS).issubset(result)
    assert result["missing_true_classes"] == []
    assert result["missing_predicted_classes"] == ["Diamond", "Silver"]
    assert result["zero_division"] == 0
    assert result["Balanced_Accuracy"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    "true,predicted,message",
    [([], [], "empty"), (["Bronze"], [], "same length"), ([["Bronze"]], [["Bronze"]], "one-dimensional"), (["unknown"], ["unknown"], "outside")],
)
def test_classification_validates_input(true, predicted, message):
    with pytest.raises(ValueError, match=message):
        evaluate_classification(true, predicted)


def test_single_predicted_class_has_no_nan_metrics():
    result = evaluate_classification(["Diamond", "Gold", "Silver", "Bronze"], ["Bronze"] * 4)
    for metric in COMMON_METRICS:
        assert np.isfinite(result[metric])
    assert result["missing_predicted_classes"] == ["Diamond", "Gold", "Silver"]


def test_regression_is_converted_to_ranks_with_train_thresholds():
    result = evaluate_regression(
        [0, 100, 500, 1500],
        [0, 200, 1600, -10],
        RankThresholds(q50=500, q85=1500, reference_count=10),
    )
    assert result["evaluation_target"] == "ranked FutureSpend"
    assert result["prediction_negative_values_clipped"] == 1
    assert result["Accuracy"] == pytest.approx(0.5)


def test_regression_rejects_invalid_actual_values_and_missing_thresholds():
    with pytest.raises(ValueError, match="negative actual"):
        evaluate_regression([-1], [0], (1, 2))
    with pytest.raises(ValueError, match="same length"):
        evaluate_regression([1, 2], [1], (1, 2))
    with pytest.raises(ValueError, match="thresholds"):
        evaluate_regression([1], [1], None)


def test_clustering_diagnostics_are_separate_and_handle_undefined_silhouette():
    result = evaluate_clustering([[0.0], [1.0], [2.0]], [0, 0, 0], inertia=2.0)
    assert result["Silhouette_Score"] is None
    assert result["Status"] == "silhouette_not_defined_for_cluster_count"
    assert result["Inertia"] == 2.0
    with pytest.raises(ValueError, match="match"):
        evaluate_clustering([[0.0], [1.0]], [0])


def test_model_result_does_not_fabricate_missing_metrics(tmp_path):
    result = make_model_result("Logistic Regression", "Classification", status="not_trained")
    assert all(result[key] is None for key in COMMON_METRICS)
    path = tmp_path / "ranking_metrics.csv"
    saved = write_model_results([result], path)
    assert saved.loc[0, "Accuracy"] is None or pd.isna(saved.loc[0, "Accuracy"])
    with pytest.raises(FileExistsError):
        write_model_results([result], path)
