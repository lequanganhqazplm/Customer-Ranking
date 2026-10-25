"""Module for CustomerRanking project.

Provides feature‑engineering utilities for the K‑Means pipeline.
"""
import pandas as pd
from pathlib import Path

def build_customer_features(df: pd.DataFrame, prediction_days: int = 90) -> pd.DataFrame:
    """Create per‑customer RFM (and auxiliary) features.

    The function follows the specification in *HUONG_DAN_TEAM.md*:

    1. **Prediction window** – the last ``prediction_days`` days of the dataset.
       ``prediction_start`` is computed as ``max(InvoiceDate) - prediction_days``.
    2. Only **historical transactions** (``InvoiceDate < prediction_start``) are used to
       compute the RFM features.
    3. Features returned (one row per ``CustomerID``):
       - ``Recency`` – days from the most recent purchase in the historic period to
         ``prediction_start``.
       - ``Frequency`` – number of distinct invoices (orders) the customer placed.
       - ``Monetary`` – total spend (sum of ``TotalPrice``) in the historic period.
       - ``TotalQuantity`` – sum of ``Quantity``.
       - ``UniqueProducts`` – count of distinct ``StockCode`` purchased.
       - ``AvgOrderValue`` – ``Monetary`` divided by ``Frequency`` (average value per order).

    The function does **not** include target columns (``FutureSpend`` or
    ``ReturnedNext90Days``) – those are generated elsewhere.

    The resulting DataFrame is saved to ``data/processed/customer_features.csv`` for
    reuse.
    """
    if "InvoiceDate" not in df.columns:
        raise KeyError("Column 'InvoiceDate' missing – ensure data is cleaned before feature engineering.")

    # Determine prediction window boundaries
    max_date = df["InvoiceDate"].max()
    prediction_start = max_date - pd.Timedelta(days=prediction_days)

    # Historical data only
    hist = df[df["InvoiceDate"] < prediction_start]
    if hist.empty:
        raise ValueError("No historical data available before the prediction window.")

    # Group by CustomerID
    grp = hist.groupby("CustomerID")

    # Recency: days between prediction_start and last purchase date
    recency = (prediction_start - grp["InvoiceDate"].max()).dt.days

    # Frequency: number of distinct InvoiceNo
    frequency = grp["InvoiceNo"].nunique()

    # Monetary: sum of TotalPrice
    monetary = grp["TotalPrice"].sum()

    # TotalQuantity
    total_quantity = grp["Quantity"].sum()

    # UniqueProducts
    unique_products = grp["StockCode"].nunique()

    # AvgOrderValue
    # Avoid division by zero – Frequency will be >=1 for customers in historical data
    avg_order_value = monetary / frequency

    features = pd.DataFrame({
        "CustomerID": recency.index,
        "Recency": recency.values,
        "Frequency": frequency.values,
        "Monetary": monetary.values,
        "TotalQuantity": total_quantity.values,
        "UniqueProducts": unique_products.values,
        "AvgOrderValue": avg_order_value.values,
    })

    # Save to processed folder
    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / "data" / "processed" / "customer_features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)
    return features

