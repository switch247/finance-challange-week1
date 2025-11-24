import pytest
import pandas as pd
from src.fnsa.features.text_analysis import get_common_phrases, perform_topic_modeling

def test_get_common_phrases():
    df = pd.DataFrame({'text': ['apple banana', 'apple orange', 'banana']})
    phrases = get_common_phrases(df, 'text', n=2)
    # apple: 2, banana: 2, orange: 1
    # Note: 'apple' and 'banana' are tied. Order might vary slightly but counts should be correct.
    assert len(phrases) == 2
    words = [p[0] for p in phrases]
    assert 'apple' in words
    assert 'banana' in words

def test_perform_topic_modeling():
    # Need enough data for NMF
    texts = [
        "stock market rises",
        "stock market falls",
        "financial news report",
        "market analysis today",
        "stock price update"
    ]
    df = pd.DataFrame({'text': texts})
    topics = perform_topic_modeling(df, 'text', n_topics=2, n_top_words=2)
    assert len(topics) == 2
    assert isinstance(topics[0], list)
