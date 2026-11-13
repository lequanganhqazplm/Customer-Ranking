"""Random Forest classifier for the four customer ranks."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from src.feature_engineering import FEATURE_COLUMNS, RANK_ORDER


class RandomForestRankClassifier(ClassifierMixin, BaseEstimator):
    """Fit a regularized Random Forest on agreed historical customer features."""

    def __init__(
        self,
        n_estimators=100,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        bootstrap=True,
        class_weight=None,
        random_state=42,
        n_jobs=-1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        features = _validate_features(X)
        labels = _validate_labels(y, len(features))
        self.feature_names_in_ = np.asarray(FEATURE_COLUMNS, dtype=object)
        self.n_features_in_ = len(FEATURE_COLUMNS)
        self.estimator_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.estimator_.fit(features, labels)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        features = self._validated_prediction_input(X)
        predictions = self.estimator_.predict(features)
        invalid = sorted(set(predictions) - set(RANK_ORDER))
        if invalid:
            raise ValueError(f"Random Forest predicted ranks outside the four customer ranks: {invalid}")
        return predictions

    def predict_proba(self, X):
        return self.estimator_.predict_proba(self._validated_prediction_input(X))

    @property
    def feature_importances_(self):
        if not hasattr(self, "estimator_"):
            raise AttributeError("RandomForestRankClassifier must be fitted before reading feature importances.")
        return self.estimator_.feature_importances_

    def _validated_prediction_input(self, X):
        if not hasattr(self, "estimator_"):
            raise ValueError("RandomForestRankClassifier must be fitted before prediction.")
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
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
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
