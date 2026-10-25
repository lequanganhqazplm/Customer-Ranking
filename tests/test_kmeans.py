"""Basic tests for the K-Means pipeline.

Run from the project root:
    python -m pytest tests/test_kmeans.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path so ``src`` is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import clean_transactions
from src.feature_engineering import build_customer_features
from src.models.kmeans_model import train_kmeans, predict_clusters, evaluate_k_range


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_transactions():
    """Return a small synthetic transaction DataFrame for unit testing."""
    dates = pd.date_range("2011-01-01", periods=60, freq="D")
    rows = []
    for i in range(120):
        rows.append({
            "InvoiceNo": str(500000 + i // 2),
            "StockCode": f"SKU{i % 5}",
            "Description": "Test item",
            "Quantity": np.random.randint(1, 10),
            "InvoiceDate": dates[i % len(dates)],
            "UnitPrice": round(np.random.uniform(1, 50), 2),
            "CustomerID": float(10000 + i % 10),
            "Country": "United Kingdom",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_scaled_data():
    """Return a small scaled 2-D array suitable for K-Means."""
    rng = np.random.RandomState(42)
    return rng.randn(50, 3)


# ── Tests: preprocessing ───────────────────────────────────────────────────

class TestCleanTransactions:
    def test_removes_missing_customer_id(self, sample_transactions):
        df = sample_transactions.copy()
        df.loc[0, "CustomerID"] = np.nan
        cleaned = clean_transactions(df)
        assert cleaned["CustomerID"].isna().sum() == 0

    def test_removes_cancelled_invoices(self, sample_transactions):
        df = sample_transactions.copy()
        df.loc[0, "InvoiceNo"] = "C12345"
        cleaned = clean_transactions(df)
        assert not cleaned["InvoiceNo"].astype(str).str.startswith("C").any()

    def test_removes_non_positive_quantity(self, sample_transactions):
        df = sample_transactions.copy()
        df.loc[1, "Quantity"] = -5
        df.loc[2, "Quantity"] = 0
        cleaned = clean_transactions(df)
        assert (cleaned["Quantity"] > 0).all()

    def test_removes_non_positive_price(self, sample_transactions):
        df = sample_transactions.copy()
        df.loc[3, "UnitPrice"] = 0
        cleaned = clean_transactions(df)
        assert (cleaned["UnitPrice"] > 0).all()

    def test_creates_total_price(self, sample_transactions):
        cleaned = clean_transactions(sample_transactions)
        assert "TotalPrice" in cleaned.columns
        expected = cleaned["Quantity"] * cleaned["UnitPrice"]
        pd.testing.assert_series_equal(cleaned["TotalPrice"], expected,
                                       check_names=False)


# ── Tests: feature engineering ─────────────────────────────────────────────

class TestBuildCustomerFeatures:
    def test_output_columns(self, sample_transactions):
        cleaned = clean_transactions(sample_transactions)
        feats = build_customer_features(cleaned, prediction_days=20)
        for col in ("CustomerID", "Recency", "Frequency", "Monetary",
                     "TotalQuantity", "UniqueProducts", "AvgOrderValue"):
            assert col in feats.columns, f"Missing column: {col}"

    def test_one_row_per_customer(self, sample_transactions):
        cleaned = clean_transactions(sample_transactions)
        feats = build_customer_features(cleaned, prediction_days=20)
        assert feats["CustomerID"].is_unique

    def test_rfm_non_negative(self, sample_transactions):
        cleaned = clean_transactions(sample_transactions)
        feats = build_customer_features(cleaned, prediction_days=20)
        assert (feats["Recency"] >= 0).all()
        assert (feats["Frequency"] > 0).all()
        assert (feats["Monetary"] > 0).all()


# ── Tests: K-Means model ──────────────────────────────────────────────────

class TestTrainKMeans:
    def test_returns_expected_keys(self, sample_scaled_data):
        result = train_kmeans(sample_scaled_data, n_clusters=3)
        assert set(result.keys()) == {"model", "labels", "silhouette", "inertia"}

    def test_labels_shape(self, sample_scaled_data):
        result = train_kmeans(sample_scaled_data, n_clusters=3)
        assert result["labels"].shape == (sample_scaled_data.shape[0],)

    def test_correct_number_of_clusters(self, sample_scaled_data):
        for k in (2, 4):
            result = train_kmeans(sample_scaled_data, n_clusters=k)
            assert len(np.unique(result["labels"])) == k

    def test_silhouette_in_valid_range(self, sample_scaled_data):
        result = train_kmeans(sample_scaled_data, n_clusters=3)
        assert -1 <= result["silhouette"] <= 1

    def test_inertia_positive(self, sample_scaled_data):
        result = train_kmeans(sample_scaled_data, n_clusters=3)
        assert result["inertia"] > 0

    def test_too_few_samples_raises(self):
        X_tiny = np.array([[1, 2, 3]])
        with pytest.raises(ValueError):
            train_kmeans(X_tiny, n_clusters=3)


class TestPredictClusters:
    def test_predict_shape(self, sample_scaled_data):
        result = train_kmeans(sample_scaled_data, n_clusters=3)
        preds = predict_clusters(result["model"], sample_scaled_data)
        assert preds.shape == (sample_scaled_data.shape[0],)


class TestEvaluateKRange:
    def test_returns_lists_of_correct_length(self, sample_scaled_data):
        ev = evaluate_k_range(sample_scaled_data, k_min=2, k_max=5)
        assert len(ev["k_values"]) == 4
        assert len(ev["inertias"]) == 4
        assert len(ev["silhouettes"]) == 4

    def test_inertia_decreases(self, sample_scaled_data):
        ev = evaluate_k_range(sample_scaled_data, k_min=2, k_max=5)
        # Inertia should generally decrease as K increases
        assert ev["inertias"][0] >= ev["inertias"][-1]
