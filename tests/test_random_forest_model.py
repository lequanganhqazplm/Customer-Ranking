"""Unit tests for the Random Forest customer rank estimator."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.feature_engineering import FEATURE_COLUMNS
from src.models.random_forest_model import RandomForestRankClassifier


def sample_features(rows=80):
    rng = np.random.default_rng(7)
    return pd.DataFrame(rng.normal(size=(rows, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)


def sample_labels(rows=80):
    return np.array([("Diamond", "Gold", "Silver", "Bronze")[i % 4] for i in range(rows)])


def test_default_random_forest_and_fit_predict_interfaces():
    model = RandomForestRankClassifier(n_jobs=1)
    assert model.n_estimators == 100
    assert model.max_depth == 5
    assert model.min_samples_split == 20
    assert model.min_samples_leaf == 10
    assert isinstance(model.fit(sample_features(), sample_labels()).estimator_, RandomForestClassifier)
    assert set(model.predict(sample_features())) == {"Diamond", "Gold", "Silver", "Bronze"}
    assert model.predict_proba(sample_features()).shape == (80, 4)
    assert len(model.feature_importances_) == len(FEATURE_COLUMNS)


def test_model_rejects_wrong_feature_schema_and_invalid_labels():
    features = sample_features()
    model = RandomForestRankClassifier(n_estimators=2, n_jobs=1)
    with pytest.raises(ValueError, match="ordered features"):
        model.fit(features[list(reversed(FEATURE_COLUMNS))], sample_labels())
    labels = sample_labels()
    labels[0] = "VIP"
    with pytest.raises(ValueError, match="outside the four"):
        model.fit(features, labels)


def test_model_requires_fit_before_prediction():
    with pytest.raises(ValueError, match="must be fitted"):
        RandomForestRankClassifier(n_estimators=2, n_jobs=1).predict(sample_features())
