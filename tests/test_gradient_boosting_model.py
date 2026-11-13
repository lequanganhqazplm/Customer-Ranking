"""Synthetic tests for the Gradient Boosting rank classifier."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from src.feature_engineering import FEATURE_COLUMNS, RANK_ORDER
from src.models.gradient_boosting_model import GradientBoostingRankClassifier


def feature_frame(rows=80):
    values = np.arange(rows, dtype=float)
    return pd.DataFrame({column: values * (index + 1) + (values % 5) for index, column in enumerate(FEATURE_COLUMNS)})


def test_default_model_uses_gradient_boosting_without_scaling():
    model = GradientBoostingRankClassifier()
    assert isinstance(model, GradientBoostingRankClassifier)
    model.fit(feature_frame(), [RANK_ORDER[index % 4] for index in range(80)])
    assert isinstance(model.estimator_, GradientBoostingClassifier)
    assert model.estimator_.n_estimators == 100
    assert model.estimator_.learning_rate == 0.1
    assert model.estimator_.max_depth == 3
    assert model.estimator_.min_samples_split == 20
    assert model.estimator_.min_samples_leaf == 10
    assert tuple(model.feature_names_in_) == FEATURE_COLUMNS


def test_fit_predict_predict_proba_and_feature_importances():
    features = feature_frame()
    labels = [RANK_ORDER[index % 4] for index in range(len(features))]
    model = GradientBoostingRankClassifier().fit(features, labels)

    predicted = model.predict(features.iloc[:8])
    probabilities = model.predict_proba(features.iloc[:8])

    assert len(predicted) == 8
    assert set(predicted).issubset(RANK_ORDER)
    assert probabilities.shape == (8, len(model.classes_))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert model.feature_importances_.shape == (len(FEATURE_COLUMNS),)


@pytest.mark.parametrize(
    "features,labels,message",
    [
        (pd.DataFrame({"wrong": [1]}), ["Bronze"], "ordered features"),
        (feature_frame(2), ["Bronze", "VIP"], "outside the four customer ranks"),
        (feature_frame(2), ["Bronze", None], "non-missing"),
        (feature_frame(2), ["Bronze"], "match the number"),
    ],
)
def test_fit_rejects_invalid_features_or_labels(features, labels, message):
    with pytest.raises(ValueError, match=message):
        GradientBoostingRankClassifier().fit(features, labels)


def test_prediction_requires_fit_and_exact_feature_order():
    model = GradientBoostingRankClassifier()
    with pytest.raises(ValueError, match="fitted"):
        model.predict(feature_frame(2))
    model.fit(feature_frame(), [RANK_ORDER[index % 4] for index in range(80)])
    with pytest.raises(ValueError, match="ordered features"):
        model.predict(feature_frame().loc[:, list(reversed(FEATURE_COLUMNS))])
