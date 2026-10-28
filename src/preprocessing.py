"""Transaction cleaning and explicit, overwrite-protected scaler persistence."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Return valid purchase transactions without mutating the caller's frame."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input DataFrame is None or empty")
    required = {"InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    cleaned = df.copy()
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["UnitPrice"] = pd.to_numeric(cleaned["UnitPrice"], errors="coerce")
    # Canonicalize Excel numeric IDs and numeric strings before aggregation.
    # Online Retail IDs are whole numbers; malformed and fractional IDs cannot
    # safely identify a customer and are excluded rather than grouped separately.
    cleaned["CustomerID"] = pd.to_numeric(cleaned["CustomerID"], errors="coerce")
    cleaned = cleaned.loc[
        cleaned["CustomerID"].notna()
        & np.isfinite(cleaned["CustomerID"])
        & cleaned["CustomerID"].eq(np.floor(cleaned["CustomerID"]))
        & cleaned["InvoiceNo"].notna()
        & ~cleaned["InvoiceNo"].astype(str).str.strip().str.upper().str.startswith("C")
        & cleaned["InvoiceDate"].notna()
        & np.isfinite(cleaned["Quantity"])
        & np.isfinite(cleaned["UnitPrice"])
        & (cleaned["Quantity"] > 0)
        & (cleaned["UnitPrice"] > 0)
    ].copy()
    cleaned["CustomerID"] = cleaned["CustomerID"].astype(float)
    cleaned["TotalPrice"] = cleaned["Quantity"] * cleaned["UnitPrice"]
    return cleaned.reset_index(drop=True)


def fit_and_save_scaler(
    df: pd.DataFrame,
    columns: list,
    scaler_path: Path | str | None = None,
    *,
    overwrite: bool = False,
) -> StandardScaler:
    """Fit a scaler on the supplied frame; persist only to an explicit path.

    Passing ``scaler_path=None`` fits in memory and does not write the legacy
    default ``models/scaler.pkl``. A requested file write refuses to replace an
    existing artifact unless the caller explicitly opts into ``overwrite``.
    Callers must pass training rows only; this helper never splits or infers a
    train/test boundary on its own.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input DataFrame is None or empty")
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for scaler: {', '.join(missing)}")
    values = df[columns].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Scaler input must contain finite numeric values only.")

    scaler = StandardScaler().fit(values)
    if scaler_path is not None:
        path = Path(scaler_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing scaler: {path}")
        joblib.dump(scaler, path)
    return scaler
