import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
from pathlib import Path
from config.settings import settings

def train(data_path: Path = settings.RAW_DATA_DIR / settings.RAW_DATA_FILE, 
          model_path: Path = settings.MODELS_DIR / 'sentiment_model.pkl'):
    """
    Train a sentiment analysis model.
    """
    # Load and preprocess data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
        
    df = pd.read_csv(data_path)
    df.dropna(subset=['headline'], inplace=True)
    df['headline'] = df['headline'].astype(str)
    
    # Assuming 'sentiment' column exists or create dummy labels for demo (replace with actual labels)
    # For demo, let's assume binary sentiment: positive if 'up' in headline, else negative
    df['sentiment'] = df['headline'].apply(lambda x: 1 if 'up' in x.lower() else 0)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['headline'])
    y = df['sentiment']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Save model and vectorizer
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({'model': model, 'vectorizer': vectorizer}, model_path)
    print(f"Model saved to {model_path}")
    
    return model, vectorizer, X_test, y_test
