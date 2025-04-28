import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
import joblib
import os
from typing import List, Dict, Any, Tuple, Union

# Setup logging
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Class for preprocessing data before model training and inference
    """
    def __init__(
        self,
        numerical_features: List[str] = None,
        categorical_features: List[str] = None,
        target_column: str = None,
        scaler_type: str = "standard",
        handle_missing: bool = True
    ):
        """
        Initialize the preprocessor

        Args:
            numerical_features: List of numerical feature column names
            categorical_features: List of categorical feature column names
            target_column: Name of the target column
            scaler_type: Type of scaler to use ('standard' or 'minmax')
            handle_missing: Whether to handle missing values
        """
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.target_column = target_column
        self.scaler_type = scaler_type
        self.handle_missing = handle_missing
        self.preprocessor = None

        logger.info(f"Initialized DataPreprocessor with {len(self.numerical_features)} numerical features and {len(self.categorical_features)} categorical features")

    def fit(self, data: pd.DataFrame) -> 'DataPreprocessor':
        """
        Fit the preprocessor to the data

        Args:
            data: DataFrame containing the data

        Returns:
            self: The fitted preprocessor
        """
        logger.info("Fitting preprocessor to data")

        # Create preprocessing steps for numerical features
        numerical_steps = []

        if self.handle_missing:
            numerical_steps.append(('imputer', SimpleImputer(strategy='median')))

        if self.scaler_type == 'standard':
            numerical_steps.append(('scaler', StandardScaler()))
        elif self.scaler_type == 'minmax':
            numerical_steps.append(('scaler', MinMaxScaler()))

        numerical_transformer = Pipeline(steps=numerical_steps)

        # Create preprocessing steps for categorical features
        categorical_steps = []

        if self.handle_missing:
            categorical_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))

        categorical_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore')))

        categorical_transformer = Pipeline(steps=categorical_steps)

        # Combine preprocessing steps
        preprocessor_steps = []

        if self.numerical_features:
            preprocessor_steps.append(('numerical', numerical_transformer, self.numerical_features))

        if self.categorical_features:
            preprocessor_steps.append(('categorical', categorical_transformer, self.categorical_features))

        # Create and fit the preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=preprocessor_steps,
            remainder='drop'
        )

        self.preprocessor.fit(data)

        logger.info("Preprocessor fitted successfully")

        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform the data using the fitted preprocessor

        Args:
            data: DataFrame containing the data

        Returns:
            np.ndarray: The transformed data
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted yet")

        logger.info("Transforming data")

        # Check for missing values if handle_missing is False
        if not self.handle_missing:
            # Check numerical features for missing values
            for feature in self.numerical_features:
                if feature in data.columns and data[feature].isna().any():
                    raise ValueError(f"Missing values found in numerical feature '{feature}' but handle_missing=False")

            # Check categorical features for missing values
            for feature in self.categorical_features:
                if feature in data.columns and data[feature].isna().any():
                    raise ValueError(f"Missing values found in categorical feature '{feature}' but handle_missing=False")

        return self.preprocessor.transform(data)

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Fit the preprocessor to the data and transform it

        Args:
            data: DataFrame containing the data

        Returns:
            np.ndarray: The transformed data
        """
        return self.fit(data).transform(data)

    def save(self, path: str) -> None:
        """
        Save the preprocessor to a file

        Args:
            path: Path to save the preprocessor
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted yet")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        logger.info(f"Saving preprocessor to {path}")

        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'DataPreprocessor':
        """
        Load a preprocessor from a file

        Args:
            path: Path to load the preprocessor from

        Returns:
            DataPreprocessor: The loaded preprocessor
        """
        logger.info(f"Loading preprocessor from {path}")

        return joblib.load(path)

    def get_feature_names(self) -> List[str]:
        """
        Get the names of the features after transformation

        Returns:
            List[str]: The feature names
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted yet")

        # Get feature names from column transformer
        feature_names = []

        for name, transformer, features in self.preprocessor.transformers_:
            if name == 'remainder':
                continue

            if hasattr(transformer, 'get_feature_names_out'):
                # For scikit-learn >= 1.0
                if name == 'numerical':
                    feature_names.extend(features)
                else:
                    feature_names.extend(transformer.get_feature_names_out(features))
            elif hasattr(transformer, 'get_feature_names'):
                # For scikit-learn < 1.0
                if name == 'numerical':
                    feature_names.extend(features)
                else:
                    feature_names.extend(transformer.get_feature_names(features))
            else:
                feature_names.extend(features)

        return feature_names

def prepare_data_for_training(
    data: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DataPreprocessor]:
    """
    Prepare data for model training

    Args:
        data: DataFrame containing the data
        numerical_features: List of numerical feature column names
        categorical_features: List of categorical feature column names
        target_column: Name of the target column
        test_size: Proportion of data to use for testing
        random_state: Random state for reproducibility

    Returns:
        Tuple containing:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
            preprocessor: Fitted preprocessor
    """
    from sklearn.model_selection import train_test_split

    logger.info("Preparing data for training")

    # Split data into features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Create and fit preprocessor
    preprocessor = DataPreprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target_column=target_column
    )

    # Transform data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    logger.info(f"Data prepared: X_train shape: {X_train_transformed.shape}, X_test shape: {X_test_transformed.shape}")

    return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor