"""Module for CustomerRanking project.

Provides utility to load the raw dataset.
"""
import pandas as pd
from pathlib import Path

def load_data(path: Path | str = None) -> pd.DataFrame:
    """Load the Online Retail Excel dataset.

    Args:
        path: Optional explicit path to the Excel file. If ``None`` the function
            looks for ``data/raw/Online Retail.xlsx`` relative to the project root.

    Returns:
        pandas.DataFrame with the raw transaction data.

    Raises:
        FileNotFoundError: If the Excel file cannot be found.
    """
    if path is None:
        # Resolve the project root (directory containing this file's parent chain)
        project_root = Path(__file__).resolve().parents[1]
        default_path = project_root / "data" / "raw" / "Online Retail.xlsx"
        path = default_path
    else:
        path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Please place 'Online Retail.xlsx' in 'data/raw/'."
        )
    # read_excel automatically handles .xlsx using openpyxl (declared in requirements)
    df = pd.read_excel(path)
    return df

