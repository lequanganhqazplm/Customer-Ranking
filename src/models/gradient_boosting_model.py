"""Gradient Boosting classifier for the four customer ranks."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier

from src.feature_engineering import FEATURE_COLUMNS, RANK_ORDER


class GradientBoostingRankClassifier(ClassifierMixin, BaseEstimator):
    """Fit a Gradient Boosting classifier on the agreed historical features."""

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        features = _validate_features(X)
        labels = _validate_labels(y, len(features))
        self.feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)
        self.n_features_in_ = len(FEATURE_COLUMNS)
        self.estimator_ = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.estimator_.fit(features, labels)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        features = self._validated_prediction_input(X)
        predictions = self.estimator_.predict(features)
        invalid = sorted(set(predictions) - set(RANK_ORDER))
        if invalid:
            raise ValueError(f"Gradient Boosting predicted ranks outside the four customer ranks: {invalid}")
        return predictions

    def predict_proba(self, X):
        return self.estimator_.predict_proba(self._validated_prediction_input(X))

    @property
    def feature_importances_(self):
        if not hasattr(self, "estimator_"):
            raise AttributeError("GradientBoostingRankClassifier must be fitted before reading feature importances.")
        return self.estimator_.feature_importances_

    def _validated_prediction_input(self, X):
        if not hasattr(self, "estimator_"):
            raise ValueError("GradientBoostingRankClassifier must be fitted before prediction.")
        features = _validate_features(X)
        if tuple(features.columns) != tuple(self.feature_names_in_):
            raise ValueError("Prediction feature columns must match the fitted feature columns and order.")
        return features


def _validate_features(X):
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("X must be a non-empty DataFrame with the agreed historical features.")
    if tuple(X.columns) != tuple(FEATURE_COLUMNS):
        raise ValueError(f"X must contain exactly these ordered features: {list(FEATURE_COLUMNS)}")
    values = X.apply(pd.to_numeric, errors="coerce")
    array = values.to_numpy(dtype=float)
    if values.isna().any().any() or not np.isfinite(array).all():
        raise ValueError("Historical feature columns must contain finite numeric values.")
    return values


def _validate_labels(y, row_count):
    labels = np.asarray(y, dtype=object)
    if labels.ndim != 1 or len(labels) != row_count:
        raise ValueError("y must be one-dimensional and match the number of feature rows.")
    if len(labels) == 0 or pd.isna(labels).any():
        raise ValueError("y must contain non-missing rank labels.")
    invalid = sorted(set(labels.tolist()) - set(RANK_ORDER))
    if invalid:
        raise ValueError(f"y contains labels outside the four customer ranks: {invalid}")
    return labels
