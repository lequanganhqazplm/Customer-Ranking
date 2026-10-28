"""Common prediction adapters that return one of the four customer ranks."""

import numpy as np
import pandas as pd

from src.feature_engineering import FEATURE_COLUMNS, RANK_ORDER, assign_customer_rank


FORBIDDEN_INPUT_COLUMNS = {
    "customerid",
    "futurespend",
    "futurefrequency",
    "returnednext90days",
    "customerrank",
    "cluster",
}


def build_model_input(dataset: pd.DataFrame) -> pd.DataFrame:
    """Select the agreed historical feature columns and reject target leakage."""
    if not isinstance(dataset, pd.DataFrame) or dataset.empty:
        raise ValueError("dataset must be a non-empty DataFrame.")
    features = [column for column in FEATURE_COLUMNS if column in dataset.columns]
    missing = [column for column in FEATURE_COLUMNS if column not in dataset.columns]
    if missing:
        raise KeyError(f"Missing historical feature columns: {', '.join(missing)}")
    if len(features) != len(FEATURE_COLUMNS):
        raise RuntimeError("Unable to construct the agreed feature set.")
    result = dataset.loc[:, list(FEATURE_COLUMNS)].copy()
    values = result.apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Historical feature columns must contain finite numeric values.")
    # Target/identifier columns may coexist in dataset; they are deliberately
    # excluded by selection above, and can never be included by caller order.
    assert not set(result.columns).intersection(FORBIDDEN_INPUT_COLUMNS)
    return values


def predict_customer_rank(model, X, model_type: str, thresholds=None):
    """Return final rank predictions for classification, regression, or K-Means.

    Classification models predict rank labels directly. Regression predictions
    are clipped at zero and converted with thresholds learned from training
    spend. K-Means rank estimators return labels through their train-only
    cluster mapping.
    """
    normalized_type = str(model_type).strip().lower()
    if normalized_type in {"classification", "classifier"}:
        predictions = np.asarray(model.predict(X), dtype=object)
        if predictions.ndim != 1:
            raise ValueError("Classification predictions must be one-dimensional.")
        _validate_rank_labels(predictions)
        return predictions
    if normalized_type in {"regression", "regressor"}:
        if thresholds is None:
            raise ValueError("Regression rank conversion requires training-derived thresholds.")
        predicted_spend = np.asarray(model.predict(X), dtype=float)
        if predicted_spend.ndim != 1 or not np.isfinite(predicted_spend).all():
            raise ValueError("Regression predictions must be a finite one-dimensional array.")
        q50, q85 = _threshold_values(thresholds)
        return np.asarray(assign_customer_rank(np.maximum(predicted_spend, 0.0), q50, q85), dtype=object)
    if normalized_type in {"clustering", "clusterer", "kmeans"}:
        predictions = np.asarray(model.predict(X), dtype=object)
        if predictions.ndim != 1:
            raise ValueError("Mapped clustering predictions must be one-dimensional.")
        _validate_rank_labels(predictions)
        return predictions
    raise ValueError(f"Unsupported model_type: {model_type!r}")


def _validate_rank_labels(predictions):
    if pd.isna(predictions).any():
        raise ValueError("Predicted ranks cannot contain missing values.")
    invalid = sorted(set(predictions.tolist()) - set(RANK_ORDER))
    if invalid:
        raise ValueError(f"Predictions contain labels outside the four customer ranks: {invalid}")


def _threshold_values(thresholds):
    if hasattr(thresholds, "q50") and hasattr(thresholds, "q85"):
        q50, q85 = float(thresholds.q50), float(thresholds.q85)
    elif isinstance(thresholds, dict):
        q50, q85 = float(thresholds["q50"]), float(thresholds["q85"])
    else:
        q50, q85 = map(float, thresholds[:2])
    if not np.isfinite([q50, q85]).all() or q50 < 0 or q85 < q50:
        raise ValueError("Training thresholds must be finite, non-negative, and q85 >= q50.")
    return q50, q85
