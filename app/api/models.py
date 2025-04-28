from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Union, Any

class PredictionInput(BaseModel):
    """
    Input data for model prediction
    """
    features: List[float] = Field(..., description="List of features for prediction")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        },
        protected_namespaces=()
    )

class PredictionOutput(BaseModel):
    """
    Output data from model prediction
    """
    prediction: Union[float, int, str] = Field(..., description="Model prediction")
    probability: Optional[float] = Field(None, description="Prediction probability")
    prediction_time: float = Field(..., description="Time taken for prediction in seconds")
    model_version: str = Field(..., description="Version of the model used for prediction")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": "setosa",
                "probability": 0.95,
                "prediction_time": 0.0023,
                "model_version": "1"
            }
        },
        protected_namespaces=()
    )

class BatchPredictionInput(BaseModel):
    """
    Input data for batch prediction
    """
    instances: List[List[float]] = Field(..., description="List of feature instances for batch prediction")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "instances": [
                    [5.1, 3.5, 1.4, 0.2],
                    [6.2, 3.4, 5.4, 2.3]
                ]
            }
        },
        protected_namespaces=()
    )

class BatchPredictionOutput(BaseModel):
    """
    Output data from batch prediction
    """
    predictions: List[Union[float, int, str]] = Field(..., description="List of model predictions")
    probabilities: Optional[List[float]] = Field(None, description="List of prediction probabilities")
    prediction_time: float = Field(..., description="Time taken for batch prediction in seconds")
    model_version: str = Field(..., description="Version of the model used for prediction")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": ["setosa", "virginica"],
                "probabilities": [0.95, 0.89],
                "prediction_time": 0.0045,
                "model_version": "1"
            }
        },
        protected_namespaces=()
    )

class ModelInfo(BaseModel):
    """
    Information about the deployed model
    """
    name: str = Field(..., description="Name of the model")
    version: str = Field(..., description="Version of the model")
    creation_timestamp: str = Field(..., description="Timestamp when the model was created")
    input_schema: Dict[str, Any] = Field(..., description="Schema of the input data")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics of the model")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "iris_classifier",
                "version": "1",
                "creation_timestamp": "2023-10-15T10:30:45Z",
                "input_schema": {
                    "features": "array of 4 float values"
                },
                "performance_metrics": {
                    "accuracy": 0.95,
                    "f1_score": 0.94,
                    "precision": 0.96,
                    "recall": 0.93
                }
            }
        },
        protected_namespaces=()
    )