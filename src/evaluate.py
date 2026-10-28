"""Shared evaluation for the final four-class CustomerRank predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)

from src.feature_engineering import RANK_ORDER, RankThresholds, assign_customer_rank


COMMON_METRICS = (
    "Accuracy",
    "Precision_Macro",
    "Recall_Macro",
    "F1_Macro",
    "Balanced_Accuracy",
)
MODEL_RESULT_COLUMNS = (
    "Model",
    "ModelType",
    *COMMON_METRICS,
    "Training_Time",
    "Prediction_Time",
    "Status",
    "Evaluation_Set_ID",
)


def evaluate_classification(y_true, y_pred) -> dict:
    """Evaluate final rank labels using the same five metrics for every model.

    All four ranks are passed explicitly to macro metrics. Missing true or
    predicted classes therefore contribute zero instead of disappearing from
    the report, and their counts are returned for transparent interpretation.
    """
    true_values, pred_values = _validate_rank_vectors(y_true, y_pred)
    true_classes = set(true_values)
    predicted_classes = set(pred_values)
    result = {
        "Accuracy": float(accuracy_score(true_values, pred_values)),
        "Precision_Macro": float(
            precision_score(true_values, pred_values, labels=list(RANK_ORDER), average="macro", zero_division=0)
        ),
        "Recall_Macro": float(
            recall_score(true_values, pred_values, labels=list(RANK_ORDER), average="macro", zero_division=0)
        ),
        "F1_Macro": float(
            f1_score(true_values, pred_values, labels=list(RANK_ORDER), average="macro", zero_division=0)
        ),
        "Balanced_Accuracy": float(balanced_accuracy_score(true_values, pred_values)),
        "Status": "completed",
        "n_samples": int(len(true_values)),
        "labels": list(RANK_ORDER),
        "true_class_counts": {label: int(np.sum(true_values == label)) for label in RANK_ORDER},
        "predicted_class_counts": {label: int(np.sum(pred_values == label)) for label in RANK_ORDER},
        "missing_true_classes": [label for label in RANK_ORDER if label not in true_classes],
        "missing_predicted_classes": [label for label in RANK_ORDER if label not in predicted_classes],
        "zero_division": 0,
    }
    return result


def evaluate_regression(y_true_spend, y_pred_spend, thresholds) -> dict:
    """Evaluate regression after converting actual/predicted spend to ranks.

    ``thresholds`` must have been learned from training/reference customers.
    Negative predicted spend is clipped to zero as a no-positive-spend output;
    negative or non-finite actual values are rejected.
    """
    true_spend = _validate_spend_vector(y_true_spend, "y_true_spend")
    predicted_spend = _validate_spend_vector(y_pred_spend, "y_pred_spend", allow_negative=True)
    if len(true_spend) != len(predicted_spend):
        raise ValueError("y_true_spend and y_pred_spend must have the same length.")
    if len(true_spend) == 0:
        raise ValueError("Evaluation input cannot be empty.")
    q50, q85 = _threshold_values(thresholds)
    true_rank = assign_customer_rank(true_spend, q50, q85)
    predicted_rank = assign_customer_rank(np.maximum(predicted_spend, 0.0), q50, q85)
    result = evaluate_classification(true_rank, predicted_rank)
    result["ranking_thresholds"] = {"q50": q50, "q85": q85}
    result["prediction_negative_values_clipped"] = int(np.sum(predicted_spend < 0))
    result["evaluation_target"] = "ranked FutureSpend"
    return result


def evaluate_clustering(X, labels, inertia=None) -> dict:
    """Return K-Means-only diagnostics, separate from four-class rank metrics."""
    values = np.asarray(X, dtype=float)
    cluster_labels = np.asarray(labels)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("X must be a non-empty two-dimensional matrix.")
    if not np.isfinite(values).all():
        raise ValueError("X must contain finite values only.")
    if cluster_labels.ndim != 1 or len(cluster_labels) != len(values):
        raise ValueError("labels must be one-dimensional and match the number of rows in X.")
    if pd.isna(cluster_labels).any():
        raise ValueError("labels cannot contain missing values.")
    unique_clusters = np.unique(cluster_labels)
    if inertia is not None and (not np.isfinite(float(inertia)) or float(inertia) < 0):
        raise ValueError("inertia must be a finite non-negative number when supplied.")

    silhouette = None
    status = "completed"
    if len(unique_clusters) < 2 or len(unique_clusters) >= len(values):
        status = "silhouette_not_defined_for_cluster_count"
    else:
        silhouette = float(silhouette_score(values, cluster_labels))
    return {
        "Status": status,
        "n_samples": int(len(values)),
        "n_clusters": int(len(unique_clusters)),
        "Silhouette_Score": silhouette,
        "Inertia": None if inertia is None else float(inertia),
        "evaluation_scope": "unsupervised clustering diagnostic; not a rank-classification score",
    }


def make_model_result(
    model: str,
    model_type: str,
    metrics: dict | None = None,
    *,
    training_time=None,
    prediction_time=None,
    status: str = "completed",
    evaluation_set_id: str | None = None,
) -> dict:
    """Create one schema-stable comparison row; absent metrics remain empty."""
    if not str(model).strip() or not str(model_type).strip():
        raise ValueError("model and model_type must be non-empty strings.")
    row = {
        "Model": str(model),
        "ModelType": str(model_type),
        **{metric: None for metric in COMMON_METRICS},
        "Training_Time": _optional_nonnegative(training_time, "training_time"),
        "Prediction_Time": _optional_nonnegative(prediction_time, "prediction_time"),
        "Status": str(status),
        "Evaluation_Set_ID": evaluation_set_id,
    }
    if status == "completed" and metrics is None:
        raise ValueError("A completed model result requires evaluated metrics.")
    if metrics is not None and status == "completed":
        for metric in COMMON_METRICS:
            value = metrics.get(metric)
            if value is None or not np.isfinite(float(value)):
                raise ValueError(f"A completed model result requires a finite {metric}.")
            row[metric] = float(value)
    return row


def write_model_results(records, path: Path | str, *, overwrite: bool = False) -> pd.DataFrame:
    """Persist comparison rows with an exclusive-create default."""
    data = pd.DataFrame(list(records), columns=MODEL_RESULT_COLUMNS)
    if not data.empty:
        if data["Model"].isna().any() or data["Model"].astype(str).str.strip().eq("").any() or data["Model"].duplicated().any():
            raise ValueError("Model result rows must have one unique non-empty Model name each.")
        set_ids = data["Evaluation_Set_ID"].dropna().unique()
        if len(set_ids) > 1:
            raise ValueError("All model results in one comparison file must use the same evaluation set.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing result file: {destination}")
    data.to_csv(destination, index=False, encoding="utf-8", mode="w" if overwrite else "x")
    return data


def _validate_rank_vectors(y_true, y_pred):
    true_values = np.asarray(y_true, dtype=object)
    pred_values = np.asarray(y_pred, dtype=object)
    if true_values.ndim != 1 or pred_values.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")
    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(true_values) == 0:
        raise ValueError("Evaluation input cannot be empty.")
    for name, values in (("y_true", true_values), ("y_pred", pred_values)):
        if pd.isna(values).any():
            raise ValueError(f"{name} cannot contain missing labels.")
        invalid = sorted(set(values.tolist()) - set(RANK_ORDER))
        if invalid:
            raise ValueError(f"{name} contains labels outside the four customer ranks: {invalid}")
    return true_values, pred_values


def _validate_spend_vector(values, name, allow_negative=False):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values only.")
    if not allow_negative and (result < 0).any():
        raise ValueError(f"{name} cannot contain negative actual spend.")
    return result


def _threshold_values(thresholds):
    if isinstance(thresholds, RankThresholds):
        return thresholds.q50, thresholds.q85
    if isinstance(thresholds, dict):
        q50, q85 = thresholds.get("q50"), thresholds.get("q85")
    elif isinstance(thresholds, (tuple, list)) and len(thresholds) >= 2:
        q50, q85 = thresholds[:2]
    else:
        raise ValueError("thresholds must be RankThresholds, a q50/q85 mapping, or a two-value sequence.")
    q50, q85 = float(q50), float(q85)
    if not np.isfinite([q50, q85]).all() or q50 < 0 or q85 < q50:
        raise ValueError("Ranking thresholds must be finite, non-negative, and q85 >= q50.")
    return q50, q85


def _optional_nonnegative(value, name):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative when supplied.")
    return value
