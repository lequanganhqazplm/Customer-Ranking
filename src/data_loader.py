"""Load the source transaction workbook without modifying it."""

from pathlib import Path
from zipfile import BadZipFile

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TRANSACTION_COLUMNS = (
    "InvoiceNo",
    "StockCode",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
)


def load_data(file_path: Path | str | None = None) -> pd.DataFrame:
    """Read an Excel transaction file and validate its required columns.

    With no path, the project default ``data/raw/Online Retail.xlsx`` is used.
    The returned frame is the raw data; this function does not clean or write it.
    """
    source = Path(file_path) if file_path is not None else PROJECT_ROOT / "data" / "raw" / "Online Retail.xlsx"
    if not source.is_file():
        raise FileNotFoundError(f"Transaction Excel file does not exist: {source}")

    try:
        transactions = pd.read_excel(source)
    except (BadZipFile, ImportError, OSError, ValueError) as exc:
        raise RuntimeError(f"Could not read transaction Excel file {source}: {exc}") from exc

    missing = [column for column in REQUIRED_TRANSACTION_COLUMNS if column not in transactions.columns]
    if missing:
        raise KeyError(f"Transaction Excel file {source} is missing required columns: {', '.join(missing)}")
    return transactions
