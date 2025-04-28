#!/usr/bin/env python3
"""
Model Optimization Script

This script optimizes a trained model using quantization and ONNX conversion.
"""

import argparse
import logging
import os
import sys
import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf
from tensorflow import keras
import joblib
import time
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.core.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Optimize a trained model")

    parser.add_argument(
        "--model-uri",
        type=str,
        required=True,
        help="URI of the model to optimize (e.g., 'models:/model_name/stage')"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="optimized_models",
        help="Directory to save optimized models"
    )

    parser.add_argument(
        "--convert-to-onnx",
        action="store_true",
        help="Convert the model to ONNX format"
    )

    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize the model"
    )

    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register the optimized model in MLflow model registry"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name to register the optimized model with in MLflow model registry"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate for optimization"
    )

    parser.add_argument(
        "--num-features",
        type=int,
        default=10,
        help="Number of features for synthetic data"
    )

    return parser.parse_args()

def generate_synthetic_data(model, num_samples, num_features):
    """Generate synthetic data for model optimization"""
    logger.info(f"Generating {num_samples} synthetic samples with {num_features} features")

    # Check if model is a classifier or regressor
    is_classifier = hasattr(model, "classes_")

    if is_classifier:
        # Generate classification data
        num_classes = len(model.classes_) if hasattr(model, "classes_") else 2
        X, y = make_classification(
            n_samples=num_samples,
            n_features=num_features,
            n_informative=num_features // 2,
            n_redundant=num_features // 10,
            n_classes=num_classes,
            random_state=42
        )
    else:
        # Generate regression data
        X, y = make_regression(
            n_samples=num_samples,
            n_features=num_features,
            n_informative=num_features // 2,
            random_state=42
        )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def convert_to_onnx(model, X_test, output_path):
    """Convert a scikit-learn model to ONNX format"""
    logger.info("Converting model to ONNX format")

    try:
        # Import necessary libraries
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        # Define input type
        initial_type = [('float_input', FloatTensorType([None, X_test.shape[1]]))]

        # Convert model to ONNX
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        # Save model
        onnx.save_model(onnx_model, output_path)

        logger.info(f"ONNX model saved to {output_path}")

        # Verify the model
        logger.info("Verifying ONNX model")
        session = ort.InferenceSession(output_path)

        # Get input name
        input_name = session.get_inputs()[0].name

        # Run inference on test data
        onnx_pred = session.run(None, {input_name: X_test.astype(np.float32)})[0]

        logger.info(f"ONNX model verification successful. Output shape: {onnx_pred.shape}")

        return output_path

    except Exception as e:
        logger.error(f"Error converting model to ONNX: {str(e)}")
        raise e

def quantize_model(model, X_train, output_path):
    """Quantize a model using TensorFlow Lite"""
    logger.info("Quantizing model using TensorFlow")

    try:
        # Convert scikit-learn model to TensorFlow Keras model
        logger.info("Converting scikit-learn model to TensorFlow Keras model")

        # Create a simple Keras model that wraps the scikit-learn model
        class SklearnToKerasModel(keras.Model):
            def __init__(self, sklearn_model):
                super(SklearnToKerasModel, self).__init__()
                self.sklearn_model = sklearn_model

            def call(self, inputs):
                # Convert TensorFlow tensor to numpy array
                numpy_inputs = inputs.numpy() if hasattr(inputs, "numpy") else inputs

                # Get predictions from scikit-learn model
                predictions = self.sklearn_model.predict(numpy_inputs)

                # Convert predictions to tensor
                return tf.convert_to_tensor(predictions, dtype=tf.float32)

        # Create Keras model
        keras_model = SklearnToKerasModel(model)

        # Create a concrete function from the Keras model
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, X_train.shape[1]], dtype=tf.float32)])
        def serving_function(inputs):
            return keras_model(inputs)

        # Get concrete function
        concrete_func = serving_function.get_concrete_function()

        # Convert to TensorFlow Lite model
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

        # Enable quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Convert model
        tflite_model = converter.convert()

        # Save model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        logger.info(f"Quantized model saved to {output_path}")

        # Verify the model
        logger.info("Verifying quantized model")
        interpreter = tf.lite.Interpreter(model_path=output_path)
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test model on a single sample
        test_sample = X_train[0:1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_sample)
        interpreter.invoke()
        tflite_pred = interpreter.get_tensor(output_details[0]['index'])

        logger.info(f"Quantized model verification successful. Output shape: {tflite_pred.shape}")

        return output_path

    except Exception as e:
        logger.error(f"Error quantizing model: {str(e)}")
        raise e

def benchmark_model(original_model, onnx_path, tflite_path, X_test):
    """Benchmark the original and optimized models"""
    logger.info("Benchmarking models")

    results = {}

    # Benchmark original model
    logger.info("Benchmarking original model")
    start_time = time.time()
    original_model.predict(X_test)
    original_time = time.time() - start_time
    results["original"] = {
        "inference_time": original_time,
        "samples_per_second": len(X_test) / original_time
    }
    logger.info(f"Original model: {results['original']['inference_time']:.4f} seconds, {results['original']['samples_per_second']:.2f} samples/second")

    # Benchmark ONNX model if available
    if onnx_path and os.path.exists(onnx_path):
        logger.info("Benchmarking ONNX model")
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name

        start_time = time.time()
        session.run(None, {input_name: X_test.astype(np.float32)})
        onnx_time = time.time() - start_time
        results["onnx"] = {
            "inference_time": onnx_time,
            "samples_per_second": len(X_test) / onnx_time,
            "speedup": original_time / onnx_time
        }
        logger.info(f"ONNX model: {results['onnx']['inference_time']:.4f} seconds, {results['onnx']['samples_per_second']:.2f} samples/second, {results['onnx']['speedup']:.2f}x speedup")

    # Benchmark TFLite model if available
    if tflite_path and os.path.exists(tflite_path):
        logger.info("Benchmarking TFLite model")
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        start_time = time.time()
        for i in range(len(X_test)):
            interpreter.set_tensor(input_details[0]['index'], X_test[i:i+1].astype(np.float32))
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
        tflite_time = time.time() - start_time
        results["tflite"] = {
            "inference_time": tflite_time,
            "samples_per_second": len(X_test) / tflite_time,
            "speedup": original_time / tflite_time
        }
        logger.info(f"TFLite model: {results['tflite']['inference_time']:.4f} seconds, {results['tflite']['samples_per_second']:.2f} samples/second, {results['tflite']['speedup']:.2f}x speedup")

    return results

def optimize_model(args):
    """Optimize a trained model"""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Load the model
        logger.info(f"Loading model from {args.model_uri}")
        model = mlflow.sklearn.load_model(args.model_uri)

        # Generate synthetic data for optimization
        X_train, X_test, y_train, y_test = generate_synthetic_data(
            model=model,
            num_samples=args.num_samples,
            num_features=args.num_features
        )

        # Save original model
        original_model_path = os.path.join(args.output_dir, "original_model.joblib")
        joblib.dump(model, original_model_path)
        logger.info(f"Original model saved to {original_model_path}")

        # Paths for optimized models
        onnx_path = os.path.join(args.output_dir, "model.onnx") if args.convert_to_onnx else None
        tflite_path = os.path.join(args.output_dir, "model.tflite") if args.quantize else None

        # Convert to ONNX if requested
        if args.convert_to_onnx:
            onnx_path = convert_to_onnx(model, X_test, onnx_path)

        # Quantize model if requested
        if args.quantize:
            tflite_path = quantize_model(model, X_train, tflite_path)

        # Benchmark models
        benchmark_results = benchmark_model(model, onnx_path, tflite_path, X_test)

        # Save benchmark results
        benchmark_path = os.path.join(args.output_dir, "benchmark_results.json")
        import json
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_results, f, indent=4)
        logger.info(f"Benchmark results saved to {benchmark_path}")

        # Register optimized model if requested
        if args.register_model and args.convert_to_onnx:
            # Start MLflow run
            with mlflow.start_run() as run:
                run_id = run.info.run_id

                # Log original model path
                mlflow.log_param("original_model_uri", args.model_uri)

                # Log benchmark results
                for model_type, metrics in benchmark_results.items():
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(f"{model_type}_{metric_name}", metric_value)

                # Log ONNX model
                if onnx_path:
                    mlflow.log_artifact(onnx_path, "model")

                # Log TFLite model
                if tflite_path:
                    mlflow.log_artifact(tflite_path, "model")

                # Register model
                if args.model_name:
                    logger.info(f"Registering optimized model as {args.model_name}")

                    model_uri = f"runs:/{run_id}/model"
                    model_details = mlflow.register_model(model_uri, args.model_name)

                    # Transition model to Production stage
                    client = mlflow.tracking.MlflowClient()
                    client.transition_model_version_stage(
                        name=args.model_name,
                        version=model_details.version,
                        stage="Production"
                    )

                    logger.info(f"Optimized model {args.model_name} version {model_details.version} registered and set to Production stage")

        logger.info("Model optimization completed successfully")

    except Exception as e:
        logger.error(f"Error optimizing model: {str(e)}")
        raise e

if __name__ == "__main__":
    args = parse_args()
    optimize_model(args)