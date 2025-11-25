"""
Model Saver Module

Infrastructure for saving and loading trained models and their metadata.
"""

import os
import json
import joblib
from datetime import datetime
from typing import Any, Dict, Tuple


def save_model(model: Any, base_path: str, model_name: str, metadata: Dict[str, Any] = None) -> str:
    """
    Save a trained model and its metadata.
    
    Args:
        model: The trained model object
        base_path: Base directory to save models (e.g., 'models/trained')
        model_name: Name for the model file
        metadata: Additional metadata to save
        
    Returns:
        Path to the saved model file
    """
    # Ensure directory exists
    os.makedirs(base_path, exist_ok=True)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.joblib"
    meta_filename = f"{model_name}_{timestamp}_meta.json"
    
    file_path = os.path.join(base_path, filename)
    meta_path = os.path.join(base_path, meta_filename)
    
    # Save model
    joblib.dump(model, file_path)
    
    # Save metadata
    if metadata is None:
        metadata = {}
        
    metadata['saved_at'] = timestamp
    metadata['model_file'] = filename
    
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Model saved to {file_path}")
    print(f"Metadata saved to {meta_path}")
    
    return file_path


def load_model(file_path: str) -> Any:
    """
    Load a model from a file.
    
    Args:
        file_path: Path to the .joblib file
        
    Returns:
        The loaded model object
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
        
    return joblib.load(file_path)


def get_model_info(meta_path: str) -> Dict[str, Any]:
    """
    Load model metadata.
    
    Args:
        meta_path: Path to the _meta.json file
        
    Returns:
        Dictionary containing metadata
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        
    with open(meta_path, 'r') as f:
        return json.load(f)
