import pytest
from fastapi.testclient import TestClient
import numpy as np
import os
import sys
import joblib
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.ml.model import get_model

# Create test client
client = TestClient(app)

# Mock model for testing
class MockModel:
    def predict(self, X):
        return np.array([0])

    def predict_proba(self, X):
        return np.array([[0.8, 0.2]])

# Mock model info
mock_model_info = {
    "name": "test_model",
    "version": "1",
    "creation_timestamp": "2023-01-01T00:00:00Z",
    "input_schema": {"features": "feature1,feature2,feature3,feature4"},
    "performance_metrics": {
        "accuracy": 0.95,
        "f1_score": 0.94,
        "precision": 0.96,
        "recall": 0.93
    }
}

@pytest.fixture
def mock_get_model():
    with patch("app.api.endpoints.get_model") as mock:
        mock.return_value = (MockModel(), mock_model_info)
        yield mock

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint(mock_get_model):
    """Test the predict endpoint"""
    response = client.post(
        "/api/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]}
    )
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
    assert "prediction_time" in response.json()
    assert "model_version" in response.json()
    assert response.json()["model_version"] == "1"

def test_batch_predict_endpoint(mock_get_model):
    """Test the batch predict endpoint"""
    response = client.post(
        "/api/batch-predict",
        json={"instances": [[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]}
    )
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert "probabilities" in response.json()
    assert "prediction_time" in response.json()
    assert "model_version" in response.json()
    assert response.json()["model_version"] == "1"
    assert len(response.json()["predictions"]) == 2

def test_model_info_endpoint(mock_get_model):
    """Test the model info endpoint"""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    assert response.json()["name"] == "test_model"
    assert response.json()["version"] == "1"
    assert "creation_timestamp" in response.json()
    assert "input_schema" in response.json()
    assert "performance_metrics" in response.json()
    assert response.json()["performance_metrics"]["accuracy"] == 0.95