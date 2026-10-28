"""Create historical customer features and forward-looking ranking labels.

The customer cohort is defined by purchases before a single prediction cutoff.
Ranking thresholds are learned from the training customers returned by
``split_customer_dataset`` and are then applied unchanged to its test customers.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RANK_ORDER = ("Diamond", "Gold", "Silver", "Bronze")
FEATURE_COLUMNS = (
    "Recency",
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "UniqueProducts",
    "AvgOrderValue",
)
TARGET_COLUMNS = ("FutureSpend", "ReturnedNext90Days", "CustomerRank")
CUSTOMER_ID_COLUMN = "CustomerID"


@dataclass(frozen=True)
class RankThresholds:
    """Positive-spend quantiles learned from a designated reference set."""

    q50: float
    q85: float
    reference_count: int

    def as_dict(self):
        return {"q50": self.q50, "q85": self.q85, "reference_count": self.reference_count}


def assign_customer_rank(future_spend, q50: float, q85: float):
    """Map future spend to one of four ranks using fixed, inclusive boundaries.

    Zero, negative, missing, and non-finite spend are treated as no valid spend
    and become Bronze. For positive values, q85 is checked first (Diamond),
    then q50 (Gold), with smaller values assigned Silver. This makes both
    boundaries inclusive on their upper tier and handles q50 == q85 explicitly.
    """
    q50, q85 = _validate_thresholds(q50, q85)

    def rank_one(value):
        try:
            spend = float(value)
        except (TypeError, ValueError):
            return "Bronze"
        if not np.isfinite(spend) or spend <= 0:
            return "Bronze"
        if spend >= q85:
            return "Diamond"
        if spend >= q50:
            return "Gold"
        return "Silver"

    if np.isscalar(future_spend) or future_spend is None:
        return rank_one(future_spend)
    if isinstance(future_spend, pd.Series):
        return future_spend.map(rank_one).rename("CustomerRank")
    if isinstance(future_spend, np.ndarray):
        return np.asarray([rank_one(value) for value in future_spend.ravel()]).reshape(future_spend.shape)
    if isinstance(future_spend, Iterable):
        return [rank_one(value) for value in future_spend]
    return rank_one(future_spend)


def fit_rank_thresholds(future_spend) -> RankThresholds:
    """Fit P50/P85 on positive training spends only.

    Zero spend is reserved for Bronze and therefore does not collapse the
    spending tiers when non-returning customers are numerous. Empty or all-zero
    reference data yields (0, 0), which still ranks every customer Bronze if
    there are no positive targets.
    """
    values = pd.to_numeric(pd.Series(future_spend), errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("FutureSpend reference values must be finite and non-missing.")
    if (values < 0).any():
        raise ValueError("FutureSpend reference values cannot be negative.")
    positive = values[values > 0]
    if positive.size == 0:
        return RankThresholds(0.0, 0.0, 0)
    q50, q85 = np.quantile(positive, [0.50, 0.85]).tolist()
    return RankThresholds(float(q50), float(q85), int(positive.size))


def build_customer_features(
    df: pd.DataFrame,
    prediction_days: int = 90,
    output_path: Path | str | None = None,
    *,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Aggregate historical features using transactions before the cutoff.

    No file is written unless ``output_path`` is explicitly supplied. Explicit
    writes are exclusive by default; ``overwrite=True`` is an opt-in for callers
    that have independently protected existing artifacts.
    """
    transactions, prediction_start, _ = _prepare_transactions(df, prediction_days)
    history = transactions.loc[transactions["InvoiceDate"] < prediction_start]
    if history.empty:
        raise ValueError("No historical data available before the prediction window.")

    grouped = history.groupby("CustomerID", sort=True)
    last_purchase = grouped["InvoiceDate"].max()
    frequency = grouped["InvoiceNo"].nunique()
    monetary = grouped["TotalPrice"].sum()
    total_quantity = grouped["Quantity"].sum()
    unique_products = grouped["StockCode"].nunique()

    features = pd.DataFrame(
        {
            "CustomerID": last_purchase.index,
            "Recency": (prediction_start - last_purchase).dt.days,
            "Frequency": frequency.to_numpy(),
            "Monetary": monetary.to_numpy(),
            "TotalQuantity": total_quantity.to_numpy(),
            "UniqueProducts": unique_products.to_numpy(),
        }
    )
    features["Recency"] = features["Recency"].astype(int)
    features["AvgOrderValue"] = features["Monetary"] / features["Frequency"]
    features = features[["CustomerID", *FEATURE_COLUMNS]]
    features.attrs["prediction_start"] = prediction_start
    features.attrs["observation_end"] = transactions["InvoiceDate"].max()
    if output_path is not None:
        _write_csv(features, output_path, overwrite=overwrite)
    return features


def build_customer_dataset(
    df: pd.DataFrame,
    prediction_days: int = 90,
    output_path: Path | str | None = None,
    *,
    overwrite: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build one row per historical customer with 90-day future targets.

    ``CustomerRank`` is created with quantiles learned from the deterministic
    training-customer subset. The training pipeline must call
    ``split_customer_dataset`` and reapply those train-only thresholds; this
    also makes that contract explicit when this helper is used independently.
    Future-only customers are excluded from the cohort.
    """
    transactions, prediction_start, observation_end = _prepare_transactions(df, prediction_days)
    features = build_customer_features(
        transactions,
        prediction_days=prediction_days,
        output_path=None,
    )
    history_ids = set(features["CustomerID"])
    future = transactions.loc[
        (transactions["InvoiceDate"] >= prediction_start)
        & (transactions["InvoiceDate"] <= observation_end)
        & transactions["CustomerID"].isin(history_ids)
    ]
    future_spend = future.groupby("CustomerID", sort=True)["TotalPrice"].sum()
    dataset = features.copy()
    dataset["FutureSpend"] = dataset["CustomerID"].map(future_spend).fillna(0.0).astype(float)
    dataset["ReturnedNext90Days"] = (dataset["FutureSpend"] > 0).astype(int)

    train_positions, _ = _split_positions(len(dataset), test_size, random_state)
    train_reference = dataset.iloc[train_positions]["FutureSpend"]
    thresholds = fit_rank_thresholds(train_reference)
    dataset["CustomerRank"] = assign_customer_rank(dataset["FutureSpend"], thresholds.q50, thresholds.q85)
    dataset.attrs["prediction_start"] = prediction_start
    dataset.attrs["observation_end"] = observation_end
    dataset.attrs["rank_thresholds"] = thresholds.as_dict()
    dataset.attrs["split_random_state"] = random_state
    dataset.attrs["split_test_size"] = test_size

    if output_path is not None:
        _write_csv(dataset, output_path, overwrite=overwrite)
    return dataset


def split_customer_dataset(
    dataset: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split the one-row-per-customer table and learn rank thresholds on train.

    Returns ``(train, test, thresholds)``. Customer IDs are checked for
    uniqueness and overlap, and both returned frames receive rank labels based
    only on the training customers' positive FutureSpend distribution.
    """
    required = {"CustomerID", "FutureSpend"}
    missing = sorted(required - set(dataset.columns)) if isinstance(dataset, pd.DataFrame) else sorted(required)
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")
    if dataset.empty:
        raise ValueError("Customer dataset is empty.")
    if dataset["CustomerID"].isna().any() or dataset["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must be present and unique before splitting.")

    train_positions, test_positions = _split_positions(len(dataset), test_size, random_state)
    train = dataset.iloc[train_positions].copy()
    test = dataset.iloc[test_positions].copy()
    thresholds = fit_rank_thresholds(train["FutureSpend"])
    train["CustomerRank"] = assign_customer_rank(train["FutureSpend"], thresholds.q50, thresholds.q85)
    test["CustomerRank"] = assign_customer_rank(test["FutureSpend"], thresholds.q50, thresholds.q85)
    if set(train["CustomerID"]) & set(test["CustomerID"]):
        raise RuntimeError("Customer-level train/test split produced overlapping customer IDs.")
    return train, test, thresholds


def _prepare_transactions(df: pd.DataFrame, prediction_days: int):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input DataFrame is None or empty")
    if not isinstance(prediction_days, int) or prediction_days <= 0:
        raise ValueError("prediction_days must be a positive integer.")
    required = {"CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "StockCode"}
    if "TotalPrice" not in df.columns and not {"Quantity", "UnitPrice"}.issubset(df.columns):
        required.add("TotalPrice")
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    transactions = df.copy()
    transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"], errors="coerce")
    transactions["Quantity"] = pd.to_numeric(transactions["Quantity"], errors="coerce")
    if "TotalPrice" in transactions.columns:
        transactions["TotalPrice"] = pd.to_numeric(transactions["TotalPrice"], errors="coerce")
    else:
        transactions["UnitPrice"] = pd.to_numeric(transactions["UnitPrice"], errors="coerce")
        transactions["TotalPrice"] = transactions["Quantity"] * transactions["UnitPrice"]

    valid = (
        transactions["CustomerID"].notna()
        & transactions["InvoiceNo"].notna()
        & transactions["InvoiceDate"].notna()
        & np.isfinite(transactions["Quantity"])
        & np.isfinite(transactions["TotalPrice"])
        & (transactions["Quantity"] > 0)
        & (transactions["TotalPrice"] > 0)
    )
    transactions = transactions.loc[valid].copy()
    if transactions.empty:
        raise ValueError("No valid positive-purchase transactions are available.")
    max_date = transactions["InvoiceDate"].max()
    prediction_start = max_date - pd.Timedelta(days=prediction_days)
    return transactions, prediction_start, max_date


def _split_positions(row_count: int, test_size: float, random_state: int):
    if not 0 < float(test_size) < 1:
        raise ValueError("test_size must be between 0 and 1.")
    positions = np.arange(row_count)
    if row_count < 2:
        return positions, np.array([], dtype=int)
    train_positions, test_positions = train_test_split(
        positions, test_size=test_size, random_state=random_state, shuffle=True
    )
    return np.asarray(train_positions, dtype=int), np.asarray(test_positions, dtype=int)


def _validate_thresholds(q50: float, q85: float):
    try:
        q50, q85 = float(q50), float(q85)
    except (TypeError, ValueError) as exc:
        raise ValueError("Ranking thresholds must be finite numbers.") from exc
    if not np.isfinite([q50, q85]).all() or q50 < 0 or q85 < q50:
        raise ValueError("Ranking thresholds must be finite, non-negative, and q85 >= q50.")
    return q50, q85


def _write_csv(data: pd.DataFrame, output_path: Path | str, *, overwrite: bool):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    data.to_csv(path, index=False, encoding="utf-8", mode="w" if overwrite else "x")
