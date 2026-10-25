"""Module for CustomerRanking project.

Provides data cleaning utilities and scaler handling for K‑Means.
"""
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib

def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw transactions according to the rules in *HUONG_DAN_TEAM.md*.

    Steps:
    1. Drop rows with missing ``CustomerID``.
    2. Remove cancelled invoices – rows where ``InvoiceNo`` starts with ``"C"``.
    3. Keep only rows with ``Quantity > 0`` and ``UnitPrice > 0``.
    4. Convert ``InvoiceDate`` to ``datetime``.
    5. Add ``TotalPrice = Quantity * UnitPrice``.
    """
    # 1. Missing CustomerID
    df = df.dropna(subset=["CustomerID"])
    # 2. Cancelled invoices
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    # 3. Positive quantity & price
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    # 4. Parse dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    # Drop rows where date could not be parsed
    df = df.dropna(subset=["InvoiceDate"]) 
    # 5. TotalPrice
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df

def fit_and_save_scaler(df: pd.DataFrame, columns: list, scaler_path: Path | str = None) -> StandardScaler:
    """Fit a ``StandardScaler`` on the provided columns and persist it.

    Args:
        df: DataFrame containing the columns to scale.
        columns: List of column names to be scaled (e.g., ``["Recency","Frequency","Monetary"]``).
        scaler_path: Destination path for ``joblib.dump``. If ``None`` a default location
            ``models/scaler.pkl`` relative to the project root is used.
    Returns:
        The fitted ``StandardScaler`` instance.
    """
    scaler = StandardScaler()
    scaler.fit(df[columns])
    if scaler_path is None:
        project_root = Path(__file__).resolve().parents[1]
        scaler_path = project_root / "models" / "scaler.pkl"
    else:
        scaler_path = Path(scaler_path)
    # Ensure directory exists
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    return scaler

