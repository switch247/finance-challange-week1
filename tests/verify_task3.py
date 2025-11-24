import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join('src')))

def test_imports():
    print("Testing imports...")
    try:
        from fnsa.data.alignment import normalize_dates, merge_news_stock_data
        from fnsa.features.sentiment import analyze_headline_sentiment
        from fnsa.features.stock_metrics import calculate_daily_returns
        from fnsa.analysis.correlation import calculate_pearson_correlation
        from fnsa.models.sentiment_model import SentimentStockPredictor
        from fnsa.models.evaluator import evaluate_classification_model
        from fnsa.models.model_saver import save_model
        print("✅ All modules imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        sys.exit(1)

def test_functionality():
    print("\nTesting basic functionality...")
    
    # 1. Sentiment
    from fnsa.features.sentiment import analyze_headline_sentiment
    try:
        score = analyze_headline_sentiment("Stock market hits record high!")
        print(f"✅ Sentiment analysis working: {score}")
    except Exception as e:
        print(f"❌ Sentiment analysis failed: {e}")
        # NLTK data might be missing, which is expected before running setup
        if "Resource vader_lexicon not found" in str(e):
            print("⚠️ NLTK resources missing (expected)")

    # 2. Stock Metrics
    from fnsa.features.stock_metrics import calculate_daily_returns
    df = pd.DataFrame({'Close': [100, 102, 101, 105]})
    returns = calculate_daily_returns(df)
    if len(returns) == 4 and pd.isna(returns[0]):
        print("✅ Returns calculation working")
    else:
        print("❌ Returns calculation failed")

    # 3. Model
    from fnsa.models.sentiment_model import SentimentStockPredictor
    try:
        model = SentimentStockPredictor()
        print("✅ Model initialization working")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")

if __name__ == "__main__":
    test_imports()
    test_functionality()
