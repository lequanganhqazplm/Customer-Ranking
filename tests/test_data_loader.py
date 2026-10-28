"""Tests for safe loading of synthetic Excel transaction files."""

import pandas as pd
import pytest

from src import data_loader


def synthetic_excel_transactions():
    return pd.DataFrame(
        {
            "InvoiceNo": ["INV-0001", "INV-0002"],
            "StockCode": ["SKU-A", "SKU-B"],
            "Quantity": [2, 1],
            "InvoiceDate": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "UnitPrice": [5.0, 3.5],
            "CustomerID": [1001, 1002],
        }
    )


def test_load_data_reads_excel_and_leaves_source_file_unchanged(tmp_path):
    source = tmp_path / "synthetic.xlsx"
    expected = synthetic_excel_transactions()
    expected.to_excel(source, index=False)
    original_bytes = source.read_bytes()

    actual = data_loader.load_data(source)

    pd.testing.assert_frame_equal(actual, expected)
    assert source.read_bytes() == original_bytes


def test_load_data_uses_project_default_path(tmp_path, monkeypatch):
    source = tmp_path / "data" / "raw" / "Online Retail.xlsx"
    source.parent.mkdir(parents=True)
    expected = synthetic_excel_transactions()
    expected.to_excel(source, index=False)
    monkeypatch.setattr(data_loader, "PROJECT_ROOT", tmp_path)

    actual = data_loader.load_data()

    pd.testing.assert_frame_equal(actual, expected)


def test_load_data_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        data_loader.load_data(tmp_path / "missing.xlsx")


def test_load_data_reports_missing_required_columns(tmp_path):
    source = tmp_path / "incomplete.xlsx"
    pd.DataFrame({"InvoiceNo": ["0001"], "CustomerID": [1001]}).to_excel(source, index=False)

    with pytest.raises(KeyError, match="StockCode"):
        data_loader.load_data(source)


def test_load_data_reports_unreadable_excel_with_path(tmp_path):
    source = tmp_path / "not-an-excel.xlsx"
    source.write_text("not an Excel workbook", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Could not read transaction Excel file"):
        data_loader.load_data(source)
