"""Module for CustomerRanking project."""
from pathlib import Path
import pandas as pd

def load_data(path=None):
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data" / "raw" / "Online Retail.xlsx"
    return pd.read_excel(path)