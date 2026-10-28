"""Unit tests for Feature Engineering & Customer Ranking Labeling.

Tests:
1. Input validation (empty DataFrame, None, missing columns).
2. Time-window splitting (historic vs future).
3. 4-tier Customer Ranking assignment (Bronze, Silver, Gold, Diamond).
4. Edge cases in ranking thresholds: equal/tied thresholds (q50 == q85), exact boundary values, NaN/negative.
5. Handling customers with no future transactions (Bronze, FutureSpend=0).
6. Data leakage prevention.
7. Custom output paths (using tmp_path).
8. Ensuring tests NEVER overwrite real data in data/processed/.

Run with:
    python -m pytest tests/test_feature_engineering.py -v
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import (
    assign_customer_rank,
    build_customer_features,
    build_customer_dataset,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_transaction_data():
    """Tạo bộ dữ liệu giao dịch mẫu trải dài 120 ngày.

    Kỳ dự đoán 30 ngày cuối:
    - 90 ngày đầu (ngày 1 -> 90): Giai đoạn lịch sử
    - 30 ngày sau (ngày 91 -> 120): Giai đoạn tương lai
    """
    base_date = pd.Timestamp("2021-01-01")
    rows = []

    # Khách 1: Mua nhiều ở cả 2 kỳ (kỳ tương lai chi tiêu lớn -> Diamond)
    for day in [10, 30, 60]:
        rows.append({
            "InvoiceNo": f"INV_1_{day}", "StockCode": f"SKU_{day}", "Description": "Item",
            "Quantity": 10, "InvoiceDate": base_date + pd.Timedelta(days=day),
            "UnitPrice": 50.0, "CustomerID": 1001.0, "Country": "UK",
            "TotalPrice": 500.0,
        })
    for day in [95, 110]:
        rows.append({
            "InvoiceNo": f"INV_1_{day}", "StockCode": f"SKU_{day}", "Description": "Item",
            "Quantity": 40, "InvoiceDate": base_date + pd.Timedelta(days=day),
            "UnitPrice": 50.0, "CustomerID": 1001.0, "Country": "UK",
            "TotalPrice": 2000.0,
        })

    # Khách 2: Mua ở cả 2 kỳ (kỳ tương lai chi tiêu vừa -> Gold)
    rows.append({
        "InvoiceNo": "INV_2_20", "StockCode": "SKU_A", "Description": "Item",
        "Quantity": 5, "InvoiceDate": base_date + pd.Timedelta(days=20),
        "UnitPrice": 40.0, "CustomerID": 1002.0, "Country": "UK",
        "TotalPrice": 200.0,
    })
    rows.append({
        "InvoiceNo": "INV_2_100", "StockCode": "SKU_B", "Description": "Item",
        "Quantity": 15, "InvoiceDate": base_date + pd.Timedelta(days=100),
        "UnitPrice": 50.0, "CustomerID": 1002.0, "Country": "UK",
        "TotalPrice": 750.0,
    })

    # Khách 3: Mua ở cả 2 kỳ (kỳ tương lai chi tiêu ít -> Silver)
    rows.append({
        "InvoiceNo": "INV_3_50", "StockCode": "SKU_C", "Description": "Item",
        "Quantity": 2, "InvoiceDate": base_date + pd.Timedelta(days=50),
        "UnitPrice": 30.0, "CustomerID": 1003.0, "Country": "UK",
        "TotalPrice": 60.0,
    })
    rows.append({
        "InvoiceNo": "INV_3_105", "StockCode": "SKU_D", "Description": "Item",
        "Quantity": 2, "InvoiceDate": base_date + pd.Timedelta(days=105),
        "UnitPrice": 25.0, "CustomerID": 1003.0, "Country": "UK",
        "TotalPrice": 50.0,
    })

    # Khách 4: Chỉ mua trong kỳ lịch sử, không mua trong kỳ tương lai -> Bronze
    rows.append({
        "InvoiceNo": "INV_4_40", "StockCode": "SKU_E", "Description": "Item",
        "Quantity": 10, "InvoiceDate": base_date + pd.Timedelta(days=40),
        "UnitPrice": 20.0, "CustomerID": 1004.0, "Country": "UK",
        "TotalPrice": 200.0,
    })

    # Khách 5: Chỉ xuất hiện trong kỳ tương lai (khách mới, không có lịch sử)
    rows.append({
        "InvoiceNo": "INV_5_115", "StockCode": "SKU_F", "Description": "Item",
        "Quantity": 5, "InvoiceDate": base_date + pd.Timedelta(days=115),
        "UnitPrice": 100.0, "CustomerID": 1005.0, "Country": "UK",
        "TotalPrice": 500.0,
    })

    return pd.DataFrame(rows)


# ── Tests: assign_customer_rank & Edge Cases ────────────────────────────────

class TestAssignCustomerRank:
    def test_single_value_assignment(self):
        assert assign_customer_rank(0, q50=500, q85=1500) == "Bronze"
        assert assign_customer_rank(-10, q50=500, q85=1500) == "Bronze"
        assert assign_customer_rank(np.nan, q50=500, q85=1500) == "Bronze"
        assert assign_customer_rank(200, q50=500, q85=1500) == "Silver"
        assert assign_customer_rank(800, q50=500, q85=1500) == "Gold"
        assert assign_customer_rank(2000, q50=500, q85=1500) == "Diamond"

    def test_exact_boundaries(self):
        # Biên q50 thuộc Gold
        assert assign_customer_rank(500.0, q50=500.0, q85=1500.0) == "Gold"
        # Biên q85 thuộc Diamond
        assert assign_customer_rank(1500.0, q50=500.0, q85=1500.0) == "Diamond"
        # Giá trị sát dưới biên
        assert assign_customer_rank(499.99, q50=500.0, q85=1500.0) == "Silver"
        assert assign_customer_rank(1499.99, q50=500.0, q85=1500.0) == "Gold"

    def test_equal_thresholds(self):
        """Khi phân vị q50 == q85 (dữ liệu suy biến hoặc trùng giá trị)."""
        ranks = assign_customer_rank(pd.Series([0.0, 50.0, 100.0, 200.0]), q50=100.0, q85=100.0)
        assert list(ranks) == ["Bronze", "Silver", "Diamond", "Diamond"]

    def test_series_assignment(self):
        spends = pd.Series([0.0, 100.0, 600.0, 2500.0, np.nan, -5.0])
        ranks = assign_customer_rank(spends, q50=500, q85=1500)
        assert list(ranks) == ["Bronze", "Silver", "Gold", "Diamond", "Bronze", "Bronze"]


# ── Tests: Input Validation ────────────────────────────────────────────────

class TestFeatureEngineeringValidation:
    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match="Input DataFrame is None or empty"):
            build_customer_features(pd.DataFrame())
        with pytest.raises(ValueError, match="Input DataFrame is None or empty"):
            build_customer_dataset(pd.DataFrame())

    def test_none_dataframe_raises(self):
        with pytest.raises(ValueError, match="Input DataFrame is None or empty"):
            build_customer_features(None)
        with pytest.raises(ValueError, match="Input DataFrame is None or empty"):
            build_customer_dataset(None)

    def test_missing_columns_raises(self):
        incomplete = pd.DataFrame({"CustomerID": [1], "InvoiceNo": ["A"]})
        with pytest.raises(KeyError, match="Missing required columns"):
            build_customer_features(incomplete)
        with pytest.raises(KeyError, match="Missing required columns"):
            build_customer_dataset(incomplete)


# ── Tests: Data Protection & Custom Output Path ─────────────────────────────

class TestDataProtection:
    def test_no_output_path_does_not_modify_processed_dir(self, mock_transaction_data):
        """Khi output_path=None, hàm không được ghi đè thư mục data/processed/."""
        feat_file = PROJECT_ROOT / "data" / "processed" / "customer_features.csv"
        ds_file = PROJECT_ROOT / "data" / "processed" / "customer_dataset.csv"

        feat_mtime_before = feat_file.stat().st_mtime if feat_file.exists() else None
        ds_mtime_before = ds_file.stat().st_mtime if ds_file.exists() else None

        # Chạy với output_path=None
        _ = build_customer_features(mock_transaction_data, prediction_days=30, output_path=None)
        _ = build_customer_dataset(mock_transaction_data, prediction_days=30, output_path=None)

        feat_mtime_after = feat_file.stat().st_mtime if feat_file.exists() else None
        ds_mtime_after = ds_file.stat().st_mtime if ds_file.exists() else None

        assert feat_mtime_before == feat_mtime_after, "customer_features.csv bị ghi đè khi output_path=None!"
        assert ds_mtime_before == ds_mtime_after, "customer_dataset.csv bị ghi đè khi output_path=None!"

    def test_custom_output_path(self, mock_transaction_data, tmp_path):
        """Khi chỉ định output_path tùy chỉnh, file được lưu đúng vị trí chỉ định."""
        custom_feat = tmp_path / "custom_features.csv"
        custom_ds = tmp_path / "custom_dataset.csv"

        res_feat = build_customer_features(mock_transaction_data, prediction_days=30, output_path=custom_feat)
        res_ds = build_customer_dataset(mock_transaction_data, prediction_days=30, output_path=custom_ds)

        assert custom_feat.is_file()
        assert custom_ds.is_file()
        assert len(pd.read_csv(custom_feat)) == len(res_feat)
        assert len(pd.read_csv(custom_ds)) == len(res_ds)

    def test_custom_output_path_refuses_to_replace_an_existing_file(self, mock_transaction_data, tmp_path):
        existing = tmp_path / "customer_features.csv"
        existing.write_text("protected artifact\n", encoding="utf-8")
        before = existing.read_bytes()

        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            build_customer_features(mock_transaction_data, prediction_days=30, output_path=existing)

        assert existing.read_bytes() == before


# ── Tests: build_customer_dataset & Data Leakage ────────────────────────────

class TestBuildCustomerDataset:
    def test_dataset_columns(self, mock_transaction_data):
        df = build_customer_dataset(mock_transaction_data, prediction_days=30)
        expected_cols = [
            "CustomerID", "Recency", "Frequency", "Monetary",
            "TotalQuantity", "UniqueProducts", "AvgOrderValue",
            "FutureSpend", "ReturnedNext90Days", "CustomerRank",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Thiếu cột {col}"

    def test_historical_customers_only(self, mock_transaction_data):
        """Khách mới chỉ xuất hiện trong kỳ tương lai (CustomerID 1005) KHÔNG được có mặt."""
        df = build_customer_dataset(mock_transaction_data, prediction_days=30)
        cust_ids = set(df["CustomerID"])
        assert 1005.0 not in cust_ids, "Khách hàng mới ở kỳ tương lai bị rò rỉ vào dataset"
        assert cust_ids == {1001.0, 1002.0, 1003.0, 1004.0}

    def test_churn_customer_handling(self, mock_transaction_data):
        """Khách 1004 không mua ở tương lai phải có FutureSpend = 0, Returned = 0, Rank = Bronze."""
        df = build_customer_dataset(mock_transaction_data, prediction_days=30)
        row_1004 = df[df["CustomerID"] == 1004.0].iloc[0]
        assert row_1004["FutureSpend"] == 0.0
        assert row_1004["ReturnedNext90Days"] == 0
        assert row_1004["CustomerRank"] == "Bronze"

    def test_no_data_leakage_in_features(self, mock_transaction_data):
        """Đặc trưng X (Recency, Frequency, Monetary) không được thay đổi khi kỳ tương lai thay đổi."""
        df_orig = build_customer_dataset(mock_transaction_data, prediction_days=30)

        modified_data = mock_transaction_data.copy()
        max_date = modified_data["InvoiceDate"].max()
        cutoff = max_date - pd.Timedelta(days=30)
        future_mask = modified_data["InvoiceDate"] >= cutoff
        modified_data.loc[future_mask, "Quantity"] *= 10
        modified_data.loc[future_mask, "TotalPrice"] *= 10

        df_mod = build_customer_dataset(modified_data, prediction_days=30)

        feature_cols = ["Recency", "Frequency", "Monetary", "TotalQuantity", "UniqueProducts", "AvgOrderValue"]
        pd.testing.assert_frame_equal(df_orig[feature_cols], df_mod[feature_cols])

        # Chỉ các nhãn y mới thay đổi
        assert (df_mod.loc[df_mod["FutureSpend"] > 0, "FutureSpend"] > df_orig.loc[df_orig["FutureSpend"] > 0, "FutureSpend"]).all()

    def test_customer_rank_values(self, mock_transaction_data):
        """Các giá trị CustomerRank hợp lệ chỉ thuộc 4 hạng."""
        df = build_customer_dataset(mock_transaction_data, prediction_days=30)
        valid_ranks = {"Bronze", "Silver", "Gold", "Diamond"}
        assert set(df["CustomerRank"].unique()).issubset(valid_ranks)
        assert df["ReturnedNext90Days"].isin([0, 1]).all()
        assert (df["FutureSpend"] >= 0).all()
