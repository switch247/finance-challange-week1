import pandas as pd
from typing import Optional, Tuple


def add_headline_length(df: pd.DataFrame, column: str = 'headline') -> pd.DataFrame:
    """
    Add a column representing the length of the headline.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the headline column.

    Returns:
        pd.DataFrame: The DataFrame with the new 'headline_length' column.
    """
    df = df.copy()
    if column in df.columns:
        df[f'{column}_length'] = df[column].astype(str).apply(len)
    return df


def get_publisher_counts(df: pd.DataFrame, column: str = 'publisher') -> pd.Series:
    """
    Count the number of articles per publisher.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the publisher column.

    Returns:
        pd.Series: A Series containing the counts of articles per publisher.
    """
    if column in df.columns:
        return df[column].value_counts()
    return pd.Series()


def get_articles_per_date(df: pd.DataFrame, date_column: str = 'date', freq: str = 'D') -> pd.Series:
    """
    Count the number of articles per date (or other frequency).

    Args:
        df (pd.DataFrame): The input DataFrame.
        date_column (str): The name of the date column.
        freq (str): The frequency for resampling (e.g., 'D' for daily, 'M' for monthly).

    Returns:
        pd.Series: A Series containing the counts of articles over time.
    """
    if date_column in df.columns:
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce', utc=True)
            
        return df.set_index(date_column).resample(freq).size()
    return pd.Series()


def add_publication_time_features(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    Extract time-based features from the publication date.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        date_column (str): Date column name.
        
    Returns:
        pd.DataFrame: DataFrame with 'hour', 'day_of_week', 'month' columns added.
    """
    df = df.copy()
    if date_column in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce', utc=True)
            
        df['hour'] = df[date_column].dt.hour
        df['day_of_week'] = df[date_column].dt.day_name()
        df['month'] = df[date_column].dt.month_name()
    return df


def extract_publisher_domains(df: pd.DataFrame, publisher_column: str = 'publisher') -> pd.DataFrame:
    """
    Extract domain from publisher if it looks like an email address.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        publisher_column (str): Publisher column name.
        
    Returns:
        pd.DataFrame: DataFrame with 'publisher_domain' column added.
    """
    df = df.copy()
    if publisher_column in df.columns:
        # Simple extraction: if contains @, take part after @
        df['publisher_domain'] = df[publisher_column].apply(
            lambda x: x.split('@')[-1] if isinstance(x, str) and '@' in x else x
        )
    return df
