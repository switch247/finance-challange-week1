import pytest
import pandas as pd
import numpy as np
from src.fnsa.data.preprocessing import convert_to_datetime, handle_missing_values, remove_duplicates

def test_convert_to_datetime():
    df = pd.DataFrame({'date': ['2023-01-01', 'invalid']})
    df = convert_to_datetime(df, 'date')
    assert pd.api.types.is_datetime64_any_dtype(df['date'])
    assert pd.isna(df['date'][1])

def test_handle_missing_values_drop():
    df = pd.DataFrame({'col1': [1, np.nan, 3]})
    df = handle_missing_values(df, strategy='drop')
    assert len(df) == 2

def test_handle_missing_values_fill_mean():
    df = pd.DataFrame({'col1': [1, np.nan, 3]})
    df = handle_missing_values(df, strategy='fill_mean')
    assert df['col1'][1] == 2.0

def test_remove_duplicates():
    df = pd.DataFrame({'col1': [1, 1, 2], 'col2': [3, 3, 4]})
    df = remove_duplicates(df)
    assert len(df) == 2
