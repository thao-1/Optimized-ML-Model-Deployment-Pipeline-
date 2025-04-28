import pytest
import numpy as np
import pandas as pd
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.preprocessing import DataPreprocessor, prepare_data_for_training
from app.ml.evaluation import evaluate_classification_model, evaluate_regression_model

# Test data
@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.choice(['A', 'B', 'C'], 100),
        'target': np.random.choice([0, 1], 100)
    })
    return data

def test_data_preprocessor(sample_data):
    """Test the DataPreprocessor class"""
    # Create preprocessor
    preprocessor = DataPreprocessor(
        numerical_features=['feature1', 'feature2'],
        categorical_features=['feature3'],
        target_column='target'
    )

    # Test fit
    preprocessor.fit(sample_data)
    assert preprocessor.preprocessor is not None

    # Test transform
    transformed_data = preprocessor.transform(sample_data)
    assert transformed_data.shape[0] == 100  # Same number of samples
    assert transformed_data.shape[1] > 2  # More features due to one-hot encoding

    # Test fit_transform
    transformed_data = preprocessor.fit_transform(sample_data)
    assert transformed_data.shape[0] == 100

    # Test get_feature_names
    feature_names = preprocessor.get_feature_names()
    assert len(feature_names) > 0
    assert 'feature1' in feature_names
    assert 'feature2' in feature_names

    # Test save and load
    with tempfile.NamedTemporaryFile(suffix='.joblib') as tmp:
        preprocessor.save(tmp.name)
        loaded_preprocessor = DataPreprocessor.load(tmp.name)
        assert loaded_preprocessor.numerical_features == preprocessor.numerical_features
        assert loaded_preprocessor.categorical_features == preprocessor.categorical_features

def test_prepare_data_for_training(sample_data):
    """Test the prepare_data_for_training function"""
    X_train, X_test, y_train, y_test, preprocessor = prepare_data_for_training(
        data=sample_data,
        numerical_features=['feature1', 'feature2'],
        categorical_features=['feature3'],
        target_column='target',
        test_size=0.2,
        random_state=42
    )

    # Check shapes
    assert X_train.shape[0] == 80  # 80% of data
    assert X_test.shape[0] == 20   # 20% of data
    assert len(y_train) == 80
    assert len(y_test) == 20

    # Check preprocessor
    assert preprocessor is not None
    assert preprocessor.numerical_features == ['feature1', 'feature2']
    assert preprocessor.categorical_features == ['feature3']

# Mock model for testing
class MockClassifier:
    def predict(self, X):
        return np.array([0, 1, 0, 1])

    def predict_proba(self, X):
        return np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.9, 0.1],
            [0.4, 0.6]
        ])

class MockRegressor:
    def predict(self, X):
        return np.array([1.2, 3.4, 2.1, 5.6])

def test_evaluate_classification_model():
    """Test the evaluate_classification_model function"""
    model = MockClassifier()
    X_test = np.random.random((4, 5))
    y_test = np.array([0, 1, 0, 0])

    # Evaluate binary classification
    metrics = evaluate_classification_model(model, X_test, y_test)

    # Check metrics
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics
    assert "classification_report" in metrics

    # Check values
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1

def test_evaluate_regression_model():
    """Test the evaluate_regression_model function"""
    model = MockRegressor()
    X_test = np.random.random((4, 5))
    y_test = np.array([1.0, 3.0, 2.0, 5.0])

    # Evaluate regression
    metrics = evaluate_regression_model(model, X_test, y_test)

    # Check metrics
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics

    # Check values
    assert metrics["mse"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0
    assert metrics["r2"] <= 1