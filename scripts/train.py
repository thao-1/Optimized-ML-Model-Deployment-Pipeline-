#!/usr/bin/env python3
"""
Model Training Script

This script trains a machine learning model and logs it to MLflow.
"""

import argparse
import logging
import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import joblib
import json
from datetime import datetime

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.preprocessing import prepare_data_for_training
from app.ml.evaluation import (
    evaluate_classification_model,
    evaluate_regression_model,
    save_evaluation_results
)
from app.core.config import settings
from app.core.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train a machine learning model")

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the training data CSV file"
    )

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["classification", "regression"],
        required=True,
        help="Type of model to train"
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["random_forest", "logistic_regression", "gradient_boosting", "linear_regression"],
        required=True,
        help="Algorithm to use for training"
    )

    parser.add_argument(
        "--target-column",
        type=str,
        required=True,
        help="Name of the target column in the data"
    )

    parser.add_argument(
        "--numerical-features",
        type=str,
        required=True,
        help="Comma-separated list of numerical feature columns"
    )

    parser.add_argument(
        "--categorical-features",
        type=str,
        default="",
        help="Comma-separated list of categorical feature columns"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing"
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility"
    )

    parser.add_argument(
        "--hyperparams",
        type=str,
        default="{}",
        help="JSON string of hyperparameters for the model"
    )

    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register the model in MLflow model registry"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name to register the model with in MLflow model registry"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save model artifacts"
    )

    return parser.parse_args()

def train_model(args):
    """Train a machine learning model"""
    logger.info(f"Training {args.algorithm} model for {args.model_type}")

    # Load data
    logger.info(f"Loading data from {args.data_path}")
    data = pd.read_csv(args.data_path)

    # Parse features
    numerical_features = args.numerical_features.split(",")
    categorical_features = args.categorical_features.split(",") if args.categorical_features else []

    # Parse hyperparameters
    hyperparams = json.loads(args.hyperparams)

    # Prepare data for training
    X_train, X_test, y_train, y_test, preprocessor = prepare_data_for_training(
        data=data,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save preprocessor
    preprocessor_path = os.path.join(args.output_dir, "preprocessor.joblib")
    preprocessor.save(preprocessor_path)

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    # Set experiment
    experiment_name = settings.MLFLOW_EXPERIMENT_NAME
    mlflow.set_experiment(experiment_name)

    # Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run ID: {run_id}")

        # Create and train model
        if args.model_type == "classification":
            if args.algorithm == "random_forest":
                model = RandomForestClassifier(random_state=args.random_state, **hyperparams)
            elif args.algorithm == "logistic_regression":
                model = LogisticRegression(random_state=args.random_state, **hyperparams)
            else:
                raise ValueError(f"Unsupported algorithm for classification: {args.algorithm}")
        else:  # regression
            if args.algorithm == "gradient_boosting":
                model = GradientBoostingRegressor(random_state=args.random_state, **hyperparams)
            elif args.algorithm == "linear_regression":
                model = LinearRegression(**hyperparams)
            else:
                raise ValueError(f"Unsupported algorithm for regression: {args.algorithm}")

        # Train model
        logger.info("Training model")
        model.fit(X_train, y_train)

        # Evaluate model
        logger.info("Evaluating model")
        if args.model_type == "classification":
            metrics = evaluate_classification_model(model, X_test, y_test)

            # Log metrics to MLflow
            mlflow.log_metric("accuracy", metrics["accuracy"])
            if "precision" in metrics:
                mlflow.log_metric("precision", metrics["precision"])
                mlflow.log_metric("recall", metrics["recall"])
                mlflow.log_metric("f1_score", metrics["f1_score"])
            else:
                mlflow.log_metric("precision_macro", metrics["precision_macro"])
                mlflow.log_metric("recall_macro", metrics["recall_macro"])
                mlflow.log_metric("f1_score_macro", metrics["f1_score_macro"])
        else:  # regression
            metrics = evaluate_regression_model(model, X_test, y_test)

            # Log metrics to MLflow
            mlflow.log_metric("mse", metrics["mse"])
            mlflow.log_metric("rmse", metrics["rmse"])
            mlflow.log_metric("mae", metrics["mae"])
            mlflow.log_metric("r2", metrics["r2"])

        # Save evaluation results
        save_evaluation_results(
            metrics=metrics,
            output_dir=args.output_dir,
            model_name=args.model_name or f"{args.algorithm}_{args.model_type}",
            is_classification=(args.model_type == "classification")
        )

        # Log parameters to MLflow
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("algorithm", args.algorithm)
        mlflow.log_param("numerical_features", args.numerical_features)
        mlflow.log_param("categorical_features", args.categorical_features)
        mlflow.log_param("target_column", args.target_column)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)

        # Log hyperparameters to MLflow
        for key, value in hyperparams.items():
            mlflow.log_param(key, value)

        # Log input schema
        mlflow.set_tag("input_schema", ",".join(preprocessor.get_feature_names()))

        # Log model to MLflow
        logger.info("Logging model to MLflow")
        mlflow.sklearn.log_model(model, "model")

        # Save model locally
        model_path = os.path.join(args.output_dir, "model.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Register model if requested
        if args.register_model:
            model_name = args.model_name or f"{args.algorithm}_{args.model_type}"
            logger.info(f"Registering model as {model_name}")

            model_uri = f"runs:/{run_id}/model"
            model_details = mlflow.register_model(model_uri, model_name)

            # Transition model to Production stage
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=model_details.version,
                stage="Production"
            )

            logger.info(f"Model {model_name} version {model_details.version} registered and set to Production stage")

        logger.info("Training completed successfully")

if __name__ == "__main__":
    args = parse_args()
    train_model(args)