import pandas as pd
from typing import Union
import os


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a file into a pandas DataFrame.

    Supports CSV, JSON, Excel (.xlsx, .xls), and Parquet files based on file extension.

    Args:
        file_path (str): Path to the file to load.

    Returns:
        pd.DataFrame: The loaded data as a DataFrame.

    Raises:
        ValueError: If the file type is not supported.
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".csv":
        return pd.read_csv(file_path)
    elif file_extension == ".json":
        return pd.read_json(file_path)
    elif file_extension in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)
    elif file_extension == ".parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Supported types: .csv, .json, .xlsx, .xls, .parquet")
