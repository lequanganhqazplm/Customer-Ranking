"""Unit tests for Linear Regression customer ranking model."""

import numpy as np
import pandas as pd
import pytest

from src.evaluate import evaluate_regression
from src.feature_engineering import FEATURE_COLUMNS, RankThresholds
from src.models.linear_regression_model import LinearRegressionRankModel, train_linear_regression
from src.predict import predict_customer_rank


def test_linear_regression_fit_predict_basic():
    # Synthetic training data
    X_train = pd.DataFrame(
        {
            "Recency": [10, 50, 100, 20],
            "Frequency": [5, 2, 1, 8],
            "Monetary": [500.0, 100.0, 30.0, 1200.0],
            "TotalQuantity": [20, 5, 2, 60],
            "UniqueProducts": [4, 2, 1, 10],
            "AvgOrderValue": [100.0, 50.0, 30.0, 150.0],
        }
    )
    y_train = np.array([400.0, 80.0, 0.0, 1000.0])

    model = LinearRegressionRankModel()
    model.fit(X_train, y_train)

    assert model.thresholds_ is not None
    assert model.thresholds_.reference_count == 3  # 3 positive spends
    assert model.n_features_in_ == 6

    # Predict spend
    spend_pred = model.predict(X_train)
    assert len(spend_pred) == 4
    assert (spend_pred >= 0).all()

    # Predict rank
    ranks = model.predict_rank(X_train)
    assert len(ranks) == 4
    for r in ranks:
        assert r in ("Diamond", "Gold", "Silver", "Bronze")

    # Feature coefficients
    coefs = model.get_coefficients()
    assert len(coefs) == 6
    assert "Feature" in coefs.columns
    assert "Coefficient" in coefs.columns


def test_linear_regression_rejects_forbidden_columns():
    X_with_leakage = pd.DataFrame(
        {
            "Recency": [10, 20],
            "Frequency": [1, 2],
            "Monetary": [100.0, 200.0],
            "TotalQuantity": [5, 10],
            "UniqueProducts": [1, 2],
            "AvgOrderValue": [100.0, 100.0],
            "CustomerID": [12345, 12346],  # forbidden
        }
    )
    y = np.array([100.0, 200.0])
    model = LinearRegressionRankModel()
    with pytest.raises(ValueError, match="Target, identifier, and cluster columns cannot be model inputs"):
        model.fit(X_with_leakage, y)


def test_linear_regression_compatibility_with_predict_customer_rank():
    X_train = pd.DataFrame(
        {col: [1.0, 2.0, 3.0, 4.0] for col in FEATURE_COLUMNS}
    )
    y_train = np.array([0.0, 100.0, 300.0, 800.0])
    thresholds = RankThresholds(q50=300.0, q85=650.0, reference_count=3)

    model = train_linear_regression(X_train, y_train, thresholds=thresholds)

    # Use the common adapter predict_customer_rank
    ranks = predict_customer_rank(model, X_train, model_type="regression", thresholds=thresholds)
    assert len(ranks) == 4
    assert set(ranks).issubset({"Diamond", "Gold", "Silver", "Bronze"})

    # Evaluate with evaluate_regression
    spend_pred = model.predict(X_train)
    eval_result = evaluate_regression(y_train, spend_pred, thresholds)
    assert "Accuracy" in eval_result
    assert "F1_Macro" in eval_result

