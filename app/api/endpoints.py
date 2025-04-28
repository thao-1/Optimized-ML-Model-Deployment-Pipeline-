from fastapi import APIRouter, HTTPException, Depends
import logging
import time
import mlflow
from prometheus_client import Counter, Histogram

from app.api.models import (
    PredictionInput,
    PredictionOutput,
    BatchPredictionInput,
    BatchPredictionOutput,
    ModelInfo
)
from app.ml.model import get_model, predict
from app.core.config import settings

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Prometheus metrics
PREDICTION_COUNT = Counter(
    "prediction_count",
    "Number of predictions made",
    ["model_version", "status"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time taken for prediction",
    ["model_version"]
)

@router.post("/predict", response_model=PredictionOutput)
async def predict_endpoint(input_data: PredictionInput):
    """
    Make a prediction with the model
    """
    start_time = time.time()
    model_version = "unknown"

    try:
        # Get the model
        model, model_info = get_model()
        model_version = model_info.get("version", "unknown")

        # Make prediction
        prediction_result = predict(model, input_data.features)

        # Calculate prediction time
        prediction_time = time.time() - start_time

        # Record metrics if enabled
        if settings.ENABLE_METRICS:
            PREDICTION_COUNT.labels(model_version=model_version, status="success").inc()
            PREDICTION_LATENCY.labels(model_version=model_version).observe(prediction_time)

        # Log prediction
        logger.info(f"Prediction made with model version {model_version} in {prediction_time:.4f} seconds")

        # Return result
        return PredictionOutput(
            prediction=prediction_result["prediction"],
            probability=prediction_result.get("probability"),
            prediction_time=prediction_time,
            model_version=model_version
        )

    except Exception as e:
        # Record failure metric
        if settings.ENABLE_METRICS:
            PREDICTION_COUNT.labels(model_version=model_version, status="error").inc()

        # Log error
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/batch-predict", response_model=BatchPredictionOutput)
async def batch_predict_endpoint(input_data: BatchPredictionInput):
    """
    Make batch predictions with the model
    """
    start_time = time.time()
    model_version = "unknown"

    try:
        # Get the model
        model, model_info = get_model()
        model_version = model_info.get("version", "unknown")

        # Make predictions
        predictions = []
        probabilities = []

        for features in input_data.instances:
            prediction_result = predict(model, features)
            predictions.append(prediction_result["prediction"])
            if "probability" in prediction_result:
                probabilities.append(prediction_result["probability"])

        # Calculate prediction time
        prediction_time = time.time() - start_time

        # Record metrics if enabled
        if settings.ENABLE_METRICS:
            PREDICTION_COUNT.labels(model_version=model_version, status="success").inc(len(input_data.instances))
            PREDICTION_LATENCY.labels(model_version=model_version).observe(prediction_time)

        # Log prediction
        logger.info(f"Batch prediction with {len(input_data.instances)} instances made with model version {model_version} in {prediction_time:.4f} seconds")

        # Return result
        return BatchPredictionOutput(
            predictions=predictions,
            probabilities=probabilities if probabilities else None,
            prediction_time=prediction_time,
            model_version=model_version
        )

    except Exception as e:
        # Record failure metric
        if settings.ENABLE_METRICS:
            PREDICTION_COUNT.labels(model_version=model_version, status="error").inc(len(input_data.instances))

        # Log error
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@router.get("/model-info", response_model=ModelInfo)
async def model_info_endpoint():
    """
    Get information about the deployed model
    """
    try:
        # Get the model info
        _, model_info = get_model()

        # Return model info
        return ModelInfo(
            name=model_info.get("name", "unknown"),
            version=model_info.get("version", "unknown"),
            creation_timestamp=model_info.get("creation_timestamp", "unknown"),
            input_schema=model_info.get("input_schema", {}),
            performance_metrics=model_info.get("performance_metrics", {})
        )

    except Exception as e:
        # Log error
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")