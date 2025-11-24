import pandas as pd
from collections import Counter
import re
from typing import List, Tuple, Optional
from textblob import TextBlob


def get_common_words(df: pd.DataFrame, column: str, n: int = 10) -> List[Tuple[str, int]]:
    """
    Get the most common words in a text column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the text column.
        n (int): The number of top words to return.

    Returns:
        list: A list of tuples (word, count).
    """
    if column not in df.columns:
        return []

    text = ' '.join(df[column].astype(str).tolist()).lower()
    # Simple tokenization (remove non-alphanumeric)
    words = re.findall(r'\b\w+\b', text)
    # Filter out common stop words (basic list)
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'was', 'are', 'were', 'be', 'been', 'this', 'that', 'it', 'as', 'by', 'from'])
    words = [w for w in words if w not in stop_words and not w.isdigit()]
    
    return Counter(words).most_common(n)


def analyze_sentiment(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Perform sentiment analysis on a text column using TextBlob.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the text column.

    Returns:
        pd.DataFrame: The DataFrame with 'polarity' and 'subjectivity' columns added.
    """
    df_copy = df.copy()
    if column in df.columns:
        df_copy['polarity'] = df_copy[column].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
        df_copy['subjectivity'] = df_copy[column].astype(str).apply(lambda x: TextBlob(x).sentiment.subjectivity)
    print(df_copy.info())
    return df_copy


def calculate_correlation(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate the correlation matrix for the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list, optional): List of columns to include. If None, uses all numeric columns.

    Returns:
        pd.DataFrame: The correlation matrix.
    """
    if columns:
        return df[columns].corr()
    return df.select_dtypes(include=['number']).corr()
