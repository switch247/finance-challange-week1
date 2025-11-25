
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join('src')))

from fnsa.data.alignment import merge_news_stock_data, prepare_ml_features

def test_merge_and_prep():
    print("Testing merge and feature prep...")
    
    # Mock Stock Data
    dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
    stock_df = pd.DataFrame({
        'Date': dates.date,
        'Close': [100, 101, 102, 101, 103]
    })
    
    # Mock News Data (aggregated)
    # Only have news for 2 days
    news_data = {
        'date': [dates[0].date(), dates[2].date()],
        'stock': ['AAPL', 'AAPL'],
        'avg_sentiment': [0.5, -0.2],
        'sentiment_std': [0.1, np.nan], # NaN std if count=1
        'news_count': [5, 1],
        'pos_mean': [0.2, 0.0],
        'neg_mean': [0.0, 0.3],
        'neu_mean': [0.8, 0.7]
    }
    news_df = pd.DataFrame(news_data)
    
    print("Stock Data:")
    print(stock_df)
    print("\nNews Data:")
    print(news_df)
    
    # Merge
    merged = merge_news_stock_data(news_df, stock_df, 'AAPL', news_date_col='date', stock_date_col='Date')
    print("\nMerged Data:")
    print(merged[['Date', 'Close', 'avg_sentiment', 'sentiment_std', 'pos_mean']])
    
    # Check if NaNs are filled
    if merged['pos_mean'].isna().any():
        print("❌ Error: pos_mean still has NaNs!")
    else:
        print("✅ pos_mean filled correctly")
        
    if merged['sentiment_std'].isna().any():
        print("❌ Error: sentiment_std still has NaNs!")
    else:
        print("✅ sentiment_std filled correctly")

    # Prepare Features
    X, y = prepare_ml_features(merged)
    print("\nFeatures X:")
    print(X)
    print("\nTarget y:")
    print(y)
    
    if len(X) == 0:
        print("❌ Error: X is empty!")
    else:
        print(f"✅ Success: X has {len(X)} rows")

if __name__ == "__main__":
    test_merge_and_prep()
