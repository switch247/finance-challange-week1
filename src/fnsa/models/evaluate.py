import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from pathlib import Path
from config.settings import settings

def evaluate_model(model_path: Path = settings.MODELS_DIR / 'sentiment_model.pkl', 
                   data_path: Path = settings.RAW_DATA_DIR / settings.RAW_DATA_FILE):
    """
    Evaluate the saved model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    # Load model
    artifacts = joblib.load(model_path)
    model = artifacts['model']
    vectorizer = artifacts['vectorizer']
    
    # Load test data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

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
