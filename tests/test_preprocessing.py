import pytest
import numpy as np
import pandas as pd
import os
import sys
from unittest.mock import patch

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.preprocessing import DataPreprocessor

# Test data
@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    data = pd.DataFrame({
        'numerical1': np.random.normal(0, 1, 100),
        'numerical2': np.random.normal(0, 1, 100),
        'categorical1': np.random.choice(['A', 'B', 'C'], 100),
        'categorical2': np.random.choice(['X', 'Y', 'Z'], 100),
        'target': np.random.choice([0, 1], 100)
    })

    # Add some missing values
    data.loc[0:5, 'numerical1'] = np.nan
    data.loc[10:15, 'categorical1'] = None

    return data

def test_init():
    """Test DataPreprocessor initialization"""
    preprocessor = DataPreprocessor(
        numerical_features=['num1', 'num2'],
        categorical_features=['cat1', 'cat2'],
        target_column='target',
        scaler_type='standard',
        handle_missing=True
    )

    assert preprocessor.numerical_features == ['num1', 'num2']
    assert preprocessor.categorical_features == ['cat1', 'cat2']
    assert preprocessor.target_column == 'target'
    assert preprocessor.scaler_type == 'standard'
    assert preprocessor.handle_missing is True
    assert preprocessor.preprocessor is None

def test_fit(sample_data):
    """Test DataPreprocessor fit method"""
    preprocessor = DataPreprocessor(
        numerical_features=['numerical1', 'numerical2'],
        categorical_features=['categorical1', 'categorical2'],
        target_column='target'
    )

    # Fit the preprocessor
    result = preprocessor.fit(sample_data)

    # Check that the preprocessor is fitted
    assert preprocessor.preprocessor is not None
    assert result is preprocessor  # Should return self

def test_transform(sample_data):
    """Test DataPreprocessor transform method"""
    preprocessor = DataPreprocessor(
        numerical_features=['numerical1', 'numerical2'],
        categorical_features=['categorical1', 'categorical2'],
        target_column='target'
    )

    # Fit the preprocessor
    preprocessor.fit(sample_data)

    # Transform the data
    transformed = preprocessor.transform(sample_data)

    # Check the transformed data
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == sample_data.shape[0]
    assert transformed.shape[1] > 4  # More columns due to one-hot encoding

def test_fit_transform(sample_data):
    """Test DataPreprocessor fit_transform method"""
    preprocessor = DataPreprocessor(
        numerical_features=['numerical1', 'numerical2'],
        categorical_features=['categorical1', 'categorical2'],
        target_column='target'
    )

    # Fit and transform the data
    transformed = preprocessor.fit_transform(sample_data)

    # Check the transformed data
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == sample_data.shape[0]
    assert transformed.shape[1] > 4  # More columns due to one-hot encoding

def test_handle_missing(sample_data):
    """Test handling of missing values"""
    # Create preprocessor with missing value handling
    preprocessor_with_handling = DataPreprocessor(
        numerical_features=['numerical1', 'numerical2'],
        categorical_features=['categorical1', 'categorical2'],
        target_column='target',
        handle_missing=True
    )

    # Create preprocessor without missing value handling
    preprocessor_without_handling = DataPreprocessor(
        numerical_features=['numerical1', 'numerical2'],
        categorical_features=['categorical1', 'categorical2'],
        target_column='target',
        handle_missing=False
    )

    # Fit and transform with handling
    try:
        transformed_with_handling = preprocessor_with_handling.fit_transform(sample_data)
        assert transformed_with_handling is not None
    except Exception as e:
        pytest.fail(f"Preprocessor with missing value handling failed: {str(e)}")

    # Fit and transform without handling should raise an error
    with pytest.raises(Exception):
        preprocessor_without_handling.fit_transform(sample_data)

def test_different_scalers(sample_data):
    """Test different scaler types"""
    # Standard scaler
    preprocessor_standard = DataPreprocessor(
        numerical_features=['numerical1', 'numerical2'],
        categorical_features=['categorical1', 'categorical2'],
        target_column='target',
        scaler_type='standard'
    )

    # MinMax scaler
    preprocessor_minmax = DataPreprocessor(
        numerical_features=['numerical1', 'numerical2'],
        categorical_features=['categorical1', 'categorical2'],
        target_column='target',
        scaler_type='minmax'
    )

    # Fit and transform with standard scaler
    transformed_standard = preprocessor_standard.fit_transform(sample_data)

    # Fit and transform with minmax scaler
    transformed_minmax = preprocessor_minmax.fit_transform(sample_data)

    # Both should work, but produce different results
    assert transformed_standard is not None
    assert transformed_minmax is not None
    assert not np.array_equal(transformed_standard, transformed_minmax)

def test_get_feature_names(sample_data):
    """Test getting feature names after transformation"""
    preprocessor = DataPreprocessor(
        numerical_features=['numerical1', 'numerical2'],
        categorical_features=['categorical1', 'categorical2'],
        target_column='target'
    )

    # Fit the preprocessor
    preprocessor.fit(sample_data)

    # Get feature names
    feature_names = preprocessor.get_feature_names()

    # Check feature names
    assert isinstance(feature_names, list)
    assert len(feature_names) > 4  # More features due to one-hot encoding
    assert 'numerical1' in feature_names
    assert 'numerical2' in feature_names