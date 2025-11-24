"""
Sentiment Analysis Module using NLTK

This module provides functions for analyzing sentiment in financial news headlines
using NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner).
"""

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')


def setup_nltk_resources():
    """Download required NLTK data packages."""
    required_packages = ['vader_lexicon', 'punkt', 'stopwords']
    for package in required_packages:
        try:
            nltk.data.find(f'sentiment/{package}' if package == 'vader_lexicon' else f'tokenizers/{package}' if package == 'punkt' else f'corpora/{package}')
        except LookupError:
            print(f"Downloading {package}...")
            nltk.download(package, quiet=True)


def analyze_headline_sentiment(headline: str) -> Dict[str, float]:
    """
    Analyze sentiment of a single headline using NLTK VADER.
    
    Args:
        headline: News headline text
        
    Returns:
        Dictionary with sentiment scores:
        - neg: Negative sentiment (0-1)
        - neu: Neutral sentiment (0-1)
        - pos: Positive sentiment (0-1)
        - compound: Overall sentiment (-1 to 1)
    """
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(headline)
    return scores


def batch_sentiment_analysis(headlines: pd.Series) -> pd.DataFrame:
    """
    Process multiple headlines efficiently.
    
    Args:
        headlines: Pandas Series of headline texts
        
    Returns:
        DataFrame with columns: neg, neu, pos, compound, sentiment_label
    """
    sia = SentimentIntensityAnalyzer()
    
    results = []
    for headline in headlines:
        if pd.isna(headline):
            results.append({'neg': 0, 'neu': 1, 'pos': 0, 'compound': 0})
        else:
            results.append(sia.polarity_scores(str(headline)))
    
    df = pd.DataFrame(results)
    
    # Add sentiment label based on compound score
    df['sentiment_label'] = df['compound'].apply(
        lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral')
    )
    
    return df


def aggregate_daily_sentiment(df: pd.DataFrame, date_col: str = 'date', 
                              ticker_col: str = 'stock') -> pd.DataFrame:
    """
    Aggregate sentiment scores by date and ticker.
    
    Args:
        df: DataFrame with sentiment scores
        date_col: Name of date column
        ticker_col: Name of ticker column
        
    Returns:
        DataFrame with aggregated daily sentiment per ticker
    """
    # Group by date and ticker, calculate mean sentiment
    agg_dict = {
        'compound': ['mean', 'std', 'count'],
        'pos': 'mean',
        'neg': 'mean',
        'neu': 'mean'
    }
    
    aggregated = df.groupby([date_col, ticker_col]).agg(agg_dict).reset_index()
    
    # Flatten column names
    aggregated.columns = [
        f'{col[0]}_{col[1]}' if col[1] else col[0] 
        for col in aggregated.columns
    ]
    
    # Rename for clarity
    aggregated.rename(columns={
        'compound_mean': 'avg_sentiment',
        'compound_std': 'sentiment_std',
        'compound_count': 'news_count'
    }, inplace=True)
    
    return aggregated


def extract_sentiment_features(headline: str) -> Dict[str, any]:
    """
    Extract comprehensive features for ML models.
    
    Args:
        headline: News headline text
        
    Returns:
        Dictionary with sentiment scores and text features
    """
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(headline)
    
    # Add text-based features
    features = sentiment.copy()
    features['headline_length'] = len(headline)
    features['word_count'] = len(headline.split())
    features['has_exclamation'] = 1 if '!' in headline else 0
    features['has_question'] = 1 if '?' in headline else 0
    features['is_uppercase'] = 1 if headline.isupper() else 0
    
    return features
