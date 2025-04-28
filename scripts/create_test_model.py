#!/usr/bin/env python3
"""
Create a test model for local development
"""

import os
import sys
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_test_model():
    """Create a test model using the Iris dataset"""
    print("Creating test model...")
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Create and train model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save model
    model_path = os.path.join("models", "model.joblib")
    joblib.dump(model, model_path)
    
    print(f"Model saved to {model_path}")
    
    # Test the model
    test_sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example iris sample
    prediction = model.predict(test_sample)
    probabilities = model.predict_proba(test_sample)
    
    print(f"Test prediction: {prediction}")
    print(f"Test probabilities: {probabilities}")
    
    return model_path

if __name__ == "__main__":
    create_test_model()
