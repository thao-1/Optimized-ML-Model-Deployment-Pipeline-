import logging
import time
import os
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import numpy as np
import joblib
from typing import Dict, List, Tuple, Any, Union
import onnxruntime as ort

from app.core.config import settings

# Setup logging
logger = logging.getLogger(__name__)

# Global variables to cache the model
_model = None
_model_info = None
_onnx_session = None

def get_model() -> Tuple[Any, Dict[str, Any]]:
    """
    Get the model from MLflow model registry or local cache

    Returns:
        Tuple[Any, Dict[str, Any]]: The model and model info
    """
    global _model, _model_info, _onnx_session

    # Return cached model if available
    if _model is not None and _model_info is not None:
        return _model, _model_info

    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

        # Get the model from MLflow model registry
        if settings.USE_ONNX:
            # Load ONNX model
            logger.info(f"Loading ONNX model {settings.MODEL_NAME} from stage {settings.MODEL_STAGE}")
            model_uri = f"models:/{settings.MODEL_NAME}/{settings.MODEL_STAGE}"
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
            onnx_path = os.path.join(local_path, "model.onnx")

            # Create ONNX inference session
            _onnx_session = ort.InferenceSession(onnx_path)
            _model = None  # We don't need the actual model for ONNX inference
        else:
            # Load regular model
            logger.info(f"Loading model {settings.MODEL_NAME} from stage {settings.MODEL_STAGE}")
            _model = mlflow.pyfunc.load_model(f"models:/{settings.MODEL_NAME}/{settings.MODEL_STAGE}")

        # Get model info from MLflow
        client = mlflow.tracking.MlflowClient()
        model_details = client.get_latest_versions(settings.MODEL_NAME, stages=[settings.MODEL_STAGE])[0]

        run = client.get_run(model_details.run_id)
        metrics = run.data.metrics
        tags = run.data.tags

        # Create model info
        _model_info = {
            "name": settings.MODEL_NAME,
            "version": model_details.version,
            "creation_timestamp": model_details.creation_timestamp,
            "input_schema": {
                "features": tags.get("input_schema", "unknown")
            },
            "performance_metrics": {
                "accuracy": metrics.get("accuracy", 0.0),
                "f1_score": metrics.get("f1_score", 0.0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0)
            }
        }

        logger.info(f"Model {settings.MODEL_NAME} version {model_details.version} loaded successfully")

        return _model, _model_info

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")

        # Fallback to local model if available
        try:
            logger.info("Attempting to load local model")
            local_model_path = "models/model.joblib"

            if os.path.exists(local_model_path):
                _model = joblib.load(local_model_path)
                _model_info = {
                    "name": "local_model",
                    "version": "local",
                    "creation_timestamp": str(time.time()),
                    "input_schema": {"features": "unknown"},
                    "performance_metrics": {}
                }
                logger.info("Local model loaded successfully")
                return _model, _model_info
        except Exception as local_error:
            logger.error(f"Error loading local model: {str(local_error)}")

        # If all else fails, raise the original error
        raise e

def predict(model: Any, features: List[float]) -> Dict[str, Any]:
    """
    Make a prediction with the model

    Args:
        model: The model to use for prediction
        features: The features to use for prediction

    Returns:
        Dict[str, Any]: The prediction result
    """
    try:
        # Convert features to numpy array
        features_array = np.array(features).reshape(1, -1)

        # Make prediction
        if settings.USE_ONNX and _onnx_session is not None:
            # ONNX inference
            input_name = _onnx_session.get_inputs()[0].name
            output_name = _onnx_session.get_outputs()[0].name

            # Run inference
            prediction = _onnx_session.run([output_name], {input_name: features_array.astype(np.float32)})[0]

            # Process prediction
            if prediction.shape[1] > 1:  # Multi-class classification
                class_idx = np.argmax(prediction, axis=1)[0]
                probability = float(prediction[0, class_idx])
                return {
                    "prediction": int(class_idx),
                    "probability": probability
                }
            else:  # Regression or binary classification
                return {
                    "prediction": float(prediction[0, 0])
                }
        else:
            # Regular model inference
            prediction = model.predict(features_array)

            # Check if the model has predict_proba method (for classification)
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features_array)[0]
                class_idx = np.argmax(probabilities)
                return {
                    "prediction": prediction[0],
                    "probability": float(probabilities[class_idx])
                }
            else:
                return {
                    "prediction": prediction[0]
                }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise e