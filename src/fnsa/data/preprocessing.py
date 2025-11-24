import pandas as pd
import numpy as np
from typing import List, Optional, Union


def convert_to_datetime(df: pd.DataFrame, column: str = 'date') -> pd.DataFrame:
    """
    Convert a column to datetime objects.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to convert.

    Returns:
        pd.DataFrame: The DataFrame with the converted column.
    """
    df = df.copy()
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors='coerce', utc=True)
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop',
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        strategy (str): Strategy for handling missing values ('drop', 'fill_mean', 'fill_median', 'fill_mode', 'fill_zero').
        columns (list, optional): List of columns to apply the strategy to. If None, applies to all appropriate columns.

    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    df = df.copy()
    
    if columns is None:
        columns = df.columns.tolist()
        
    if strategy == 'drop':
        df = df.dropna(subset=columns)
    elif strategy == 'fill_mean':
        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].mean())
    elif strategy == 'fill_median':
        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
    elif strategy == 'fill_mode':
        for col in columns:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
    elif strategy == 'fill_zero':
        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
    return df


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        subset (list, optional): Subset of columns to consider for identifying duplicates.

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates(subset=subset)
