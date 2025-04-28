#!/usr/bin/env python3
"""
Model Registration Script

This script registers a model from a run to the MLflow model registry.
"""

import argparse
import logging
import os
import sys
import mlflow
from mlflow.tracking import MlflowClient

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.core.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Register a model to MLflow model registry")

    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="MLflow run ID containing the model to register"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name to register the model with in MLflow model registry"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="model",
        help="Path to the model in the MLflow run artifacts"
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=["None", "Staging", "Production", "Archived"],
        default="None",
        help="Stage to set for the registered model version"
    )

    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Description for the registered model version"
    )

    return parser.parse_args()

def register_model(args):
    """Register a model to MLflow model registry"""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    # Create MLflow client
    client = MlflowClient()

    try:
        # Check if run exists
        run = client.get_run(args.run_id)
        logger.info(f"Found run {args.run_id}")

        # Register model
        logger.info(f"Registering model {args.model_name} from run {args.run_id}")
        model_uri = f"runs:/{args.run_id}/{args.model_path}"
        model_details = mlflow.register_model(model_uri, args.model_name)

        # Set stage if specified
        if args.stage != "None":
            logger.info(f"Setting model {args.model_name} version {model_details.version} to stage {args.stage}")
            client.transition_model_version_stage(
                name=args.model_name,
                version=model_details.version,
                stage=args.stage
            )

        # Set description if specified
        if args.description:
            logger.info(f"Setting description for model {args.model_name} version {model_details.version}")
            client.update_model_version(
                name=args.model_name,
                version=model_details.version,
                description=args.description
            )

        logger.info(f"Model {args.model_name} version {model_details.version} registered successfully")

        # Get model details
        model_version = client.get_model_version(args.model_name, model_details.version)
        logger.info(f"Model version details: {model_version}")

        return model_details.version

    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")
        raise e

if __name__ == "__main__":
    args = parse_args()
    register_model(args)