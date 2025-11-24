import pytest
import pandas as pd
from src.fnsa.features.builder import (
    add_headline_length,
    get_publisher_counts,
    add_publication_time_features,
    extract_publisher_domains
)

def test_add_headline_length():
    df = pd.DataFrame({'headline': ['short', 'longer headline']})
    df = add_headline_length(df)
    assert 'headline_length' in df.columns
    assert df['headline_length'][0] == 5
    assert df['headline_length'][1] == 15

def test_get_publisher_counts():
    df = pd.DataFrame({'publisher': ['A', 'A', 'B']})
    counts = get_publisher_counts(df)
    assert counts['A'] == 2
    assert counts['B'] == 1

def test_add_publication_time_features():
    df = pd.DataFrame({'date': ['2023-01-01 10:00:00']})
    df = add_publication_time_features(df)
    assert df['hour'][0] == 10
    assert df['day_of_week'][0] == 'Sunday'

def test_extract_publisher_domains():
    df = pd.DataFrame({'publisher': ['user@example.com', 'Company']})
    df = extract_publisher_domains(df)
    assert df['publisher_domain'][0] == 'example.com'
    assert df['publisher_domain'][1] == 'Company'
