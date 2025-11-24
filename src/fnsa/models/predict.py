import joblib
import os
from pathlib import Path
from config.settings import settings

def predict(text, model_path: Path = settings.MODELS_DIR / 'sentiment_model.pkl'):
    """
    Predict sentiment for a given text.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    # Load model
    artifacts = joblib.load(model_path)
    model = artifacts['model']
    vectorizer = artifacts['vectorizer']
    
    # Transform text
    X = vectorizer.transform([text])
    
    # Predict
    prediction = model.predict(X)[0]
    return prediction
