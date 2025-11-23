# train
# save
# evaluate_model


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def train(data_path='../data/raw_analyst_ratings.csv', model_path='../models/sentiment_model.pkl'):
    # Load and preprocess data
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

def save(model, vectorizer, path='../models/sentiment_model.pkl'):
    joblib.dump({'model': model, 'vectorizer': vectorizer}, path)
    print(f"Model saved to {path}")

def evaluate_model(model_path='../models/sentiment_model.pkl', data_path='../data/raw_analyst_ratings.csv'):
    # Load model
    artifacts = joblib.load(model_path)
    model = artifacts['model']
    vectorizer = artifacts['vectorizer']
    
    # Load test data
    df = pd.read_csv(data_path)
    df.dropna(subset=['headline'], inplace=True)
    df['headline'] = df['headline'].astype(str)
    df['sentiment'] = df['headline'].apply(lambda x: 1 if 'up' in x.lower() else 0)  # Dummy labels
    
    X_test = vectorizer.transform(df['headline'])
    y_test = df['sentiment']
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))