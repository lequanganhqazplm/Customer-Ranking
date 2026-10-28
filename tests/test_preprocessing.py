"""Unit tests for data preprocessing module (src/preprocessing.py).

Tests:
1. Input validation (empty DataFrame, missing columns).
2. Type conversions (Quantity, UnitPrice, InvoiceDate).
3. Cleaning rules (missing CustomerID, cancelled invoices, non-positive values).
4. TotalPrice calculation.
5. StandardScaler fitting and persisting with custom path.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import clean_transactions, fit_and_save_scaler


@pytest.fixture
def sample_raw_data():
    """Tạo DataFrame giao dịch thô kiểm tra các trường hợp bất thường."""
    return pd.DataFrame([
        # Hợp lệ 1
        {"InvoiceNo": "536365", "StockCode": "85123A", "Description": "WHITE HANGING HEART",
         "Quantity": 6, "InvoiceDate": "2010-12-01 08:26:00", "UnitPrice": 2.55, "CustomerID": 17850.0},
        # Hợp lệ 2 (dạng chuỗi cần chuyển số)
        {"InvoiceNo": "536366", "StockCode": "22633", "Description": "HAND WARMER UNION JACK",
         "Quantity": "10", "InvoiceDate": "2010-12-01 08:28:00", "UnitPrice": "1.85", "CustomerID": 17850.0},
        # Thiếu CustomerID
        {"InvoiceNo": "536367", "StockCode": "84879", "Description": "ASSORTED COLOUR BIRD",
         "Quantity": 32, "InvoiceDate": "2010-12-01 08:34:00", "UnitPrice": 1.69, "CustomerID": np.nan},
        # Hóa đơn hủy (bắt đầu bằng C)
        {"InvoiceNo": "C536379", "StockCode": "D", "Description": "Discount",
         "Quantity": -1, "InvoiceDate": "2010-12-01 09:41:00", "UnitPrice": 27.50, "CustomerID": 14527.0},
        # Quantity âm không phải hóa đơn C
        {"InvoiceNo": "536380", "StockCode": "22960", "Description": "JAM MAKING SET",
         "Quantity": -5, "InvoiceDate": "2010-12-01 09:50:00", "UnitPrice": 4.25, "CustomerID": 15311.0},
        # Quantity = 0
        {"InvoiceNo": "536381", "StockCode": "22961", "Description": "JAM MAKING SET 2",
         "Quantity": 0, "InvoiceDate": "2010-12-01 09:55:00", "UnitPrice": 4.25, "CustomerID": 15311.0},
        # UnitPrice <= 0
        {"InvoiceNo": "536382", "StockCode": "22962", "Description": "JAM MAKING SET 3",
         "Quantity": 1, "InvoiceDate": "2010-12-01 09:58:00", "UnitPrice": 0.0, "CustomerID": 15311.0},
        # Ngày sai định dạng hoàn toàn
        {"InvoiceNo": "536383", "StockCode": "22963", "Description": "JAM MAKING SET 4",
         "Quantity": 1, "InvoiceDate": "invalid_date", "UnitPrice": 5.0, "CustomerID": 15311.0},
    ])


class TestPreprocessingValidation:
    def test_empty_dataframe_raises_value_error(self):
        with pytest.raises(ValueError, match="Input DataFrame is None or empty"):
            clean_transactions(pd.DataFrame())

    def test_none_dataframe_raises_value_error(self):
        with pytest.raises(ValueError, match="Input DataFrame is None or empty"):
            clean_transactions(None)

    def test_missing_required_columns_raises_key_error(self):
        incomplete_df = pd.DataFrame({"Quantity": [1], "UnitPrice": [10.0]})
        with pytest.raises(KeyError, match="Missing required columns"):
            clean_transactions(incomplete_df)


class TestPreprocessingCleaningLogic:
    def test_clean_transactions_filters_correctly(self, sample_raw_data):
        cleaned = clean_transactions(sample_raw_data)
        # Chỉ có 2 dòng đầu tiên là hợp lệ
        assert len(cleaned) == 2
        assert set(cleaned["InvoiceNo"]) == {"536365", "536366"}

    def test_total_price_calculation(self, sample_raw_data):
        cleaned = clean_transactions(sample_raw_data)
        assert "TotalPrice" in cleaned.columns
        expected = cleaned["Quantity"] * cleaned["UnitPrice"]
        pd.testing.assert_series_equal(cleaned["TotalPrice"], expected, check_names=False)

    def test_numeric_conversion(self, sample_raw_data):
        cleaned = clean_transactions(sample_raw_data)
        assert pd.api.types.is_numeric_dtype(cleaned["Quantity"])
        assert pd.api.types.is_numeric_dtype(cleaned["UnitPrice"])
        assert pd.api.types.is_datetime64_any_dtype(cleaned["InvoiceDate"])

    def test_customer_ids_are_normalized_and_invalid_ids_are_excluded(self):
        rows = pd.DataFrame(
            {
                "InvoiceNo": ["1", "2", "3", "4"],
                "StockCode": ["A", "A", "A", "A"],
                "Quantity": [1, 1, 1, 1],
                "InvoiceDate": ["2011-01-01"] * 4,
                "UnitPrice": [2.0, 2.0, 2.0, 2.0],
                "CustomerID": ["17850", 17850.0, "bad-id", 17850.5],
            }
        )
        original = rows.copy(deep=True)

        cleaned = clean_transactions(rows)

        assert cleaned["CustomerID"].tolist() == [17850.0, 17850.0]
        assert cleaned["CustomerID"].dtype == float
        pd.testing.assert_frame_equal(rows, original)


class TestScalerFitting:
    def test_fit_and_save_scaler_custom_path(self, tmp_path):
        custom_scaler_path = tmp_path / "custom_scaler.pkl"
        df = pd.DataFrame({
            "Recency": [10, 20, 30],
            "Frequency": [1, 2, 3],
            "Monetary": [100.0, 200.0, 300.0],
        })
        scaler = fit_and_save_scaler(df, ["Recency", "Frequency", "Monetary"], scaler_path=custom_scaler_path)
        assert custom_scaler_path.is_file()
        assert scaler.mean_ is not None

    def test_default_path_fits_in_memory_without_touching_saved_scaler(self):
        project_root = Path(__file__).resolve().parents[1]
        existing_scaler = project_root / "models" / "scaler.pkl"
        before = existing_scaler.read_bytes() if existing_scaler.exists() else None
        df = pd.DataFrame({"Recency": [1, 2, 3], "Frequency": [2, 3, 4], "Monetary": [10, 20, 30]})

        scaler = fit_and_save_scaler(df, ["Recency", "Frequency", "Monetary"])

        assert scaler.mean_ is not None
        after = existing_scaler.read_bytes() if existing_scaler.exists() else None
        assert after == before

    def test_explicit_scaler_path_refuses_existing_artifact(self, tmp_path):
        path = tmp_path / "scaler.pkl"
        path.write_bytes(b"protected")
        before = path.read_bytes()
        df = pd.DataFrame({"Recency": [1, 2], "Frequency": [2, 3], "Monetary": [10, 20]})

        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            fit_and_save_scaler(df, ["Recency", "Frequency", "Monetary"], scaler_path=path)

        assert path.read_bytes() == before
