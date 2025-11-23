from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from typing import List, Optional


def fill_missing_values(df: pd.DataFrame, numeric_strategy: str = 'mean',
                        categorical_strategy: str = 'mode') -> pd.DataFrame:
    """
    Fill missing values in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numeric_strategy (str): Strategy for filling numeric columns ('mean', 'median', 'zero').
        categorical_strategy (str): Strategy for filling categorical columns ('mode', 'unknown').

    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    if numeric_strategy == 'mean':
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)
    elif numeric_strategy == 'median':
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
    elif numeric_strategy == 'zero':
        df[numeric_cols] = df[numeric_cols].fillna(0)

    if categorical_strategy == 'mode':
        for col in categorical_cols:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col].fillna(mode_val[0], inplace=True)
    elif categorical_strategy == 'unknown':
        df[categorical_cols] = df[categorical_cols].fillna('unknown')

    return df


def preprocess_data(df: pd.DataFrame, categorical_cols: Optional[List[str]] = None,
                    fill_numeric: bool = True, numeric_strategy: str = 'mean',
                    encode_categorical: bool = True) -> pd.DataFrame:
    """
    Preprocess the DataFrame by filling missing values and encoding categorical columns.

    Args:
        df (pd.DataFrame): The input DataFrame to preprocess.
        categorical_cols (list, optional): List of column names to encode. If None, auto-detect.
        fill_numeric (bool): Whether to fill missing values in numeric columns.
        numeric_strategy (str): Strategy for filling numeric columns.
        encode_categorical (bool): Whether to encode categorical columns.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    df = df.copy()

    # Fill missing values
    if fill_numeric:
        df = fill_missing_values(df, numeric_strategy=numeric_strategy)

    # Encode categorical columns
    if encode_categorical:
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        le = LabelEncoder()
        for col in categorical_cols:
            if col in df.columns:
                df[col] = le.fit_transform(df[col].astype(str))

    return df