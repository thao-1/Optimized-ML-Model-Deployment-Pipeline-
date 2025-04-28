import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import os
from typing import Dict, List, Any, Union, Optional, Tuple

# Setup logging
logger = logging.getLogger(__name__)

def evaluate_classification_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate a classification model

    Args:
        model: The trained model
        X_test: Test features
        y_test: Test target
        class_names: Names of the classes
        threshold: Threshold for binary classification

    Returns:
        Dict[str, Any]: Dictionary of evaluation metrics
    """
    logger.info("Evaluating classification model")

    # Make predictions
    y_pred = model.predict(X_test)

    # Get probabilities if available
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = None

    # Determine if binary or multiclass
    is_binary = len(np.unique(y_test)) == 2

    # Calculate metrics
    metrics = {}

    # Basic metrics
    metrics["accuracy"] = float(accuracy_score(y_test, y_pred))

    # For binary classification
    if is_binary:
        metrics["precision"] = float(precision_score(y_test, y_pred, average='binary'))
        metrics["recall"] = float(recall_score(y_test, y_pred, average='binary'))
        metrics["f1_score"] = float(f1_score(y_test, y_pred, average='binary'))

        # ROC AUC if probabilities are available
        if y_prob is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob[:, 1]))

    # For multiclass classification
    else:
        metrics["precision_macro"] = float(precision_score(y_test, y_pred, average='macro'))
        metrics["recall_macro"] = float(recall_score(y_test, y_pred, average='macro'))
        metrics["f1_score_macro"] = float(f1_score(y_test, y_pred, average='macro'))

        metrics["precision_weighted"] = float(precision_score(y_test, y_pred, average='weighted'))
        metrics["recall_weighted"] = float(recall_score(y_test, y_pred, average='weighted'))
        metrics["f1_score_weighted"] = float(f1_score(y_test, y_pred, average='weighted'))

        # ROC AUC for multiclass if probabilities are available
        if y_prob is not None:
            try:
                metrics["roc_auc_ovr"] = float(roc_auc_score(y_test, y_prob, multi_class='ovr'))
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {str(e)}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # Classification report
    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(y_test)))]

    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    metrics["classification_report"] = report

    logger.info(f"Classification metrics: accuracy={metrics['accuracy']:.4f}")

    return metrics

def evaluate_regression_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate a regression model

    Args:
        model: The trained model
        X_test: Test features
        y_test: Test target

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    logger.info("Evaluating regression model")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {}

    metrics["mse"] = float(mean_squared_error(y_test, y_pred))
    metrics["rmse"] = float(np.sqrt(metrics["mse"]))
    metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
    metrics["r2"] = float(r2_score(y_test, y_pred))

    logger.info(f"Regression metrics: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")

    return metrics

def save_evaluation_results(
    metrics: Dict[str, Any],
    output_dir: str,
    model_name: str,
    is_classification: bool = True
) -> None:
    """
    Save evaluation results to files

    Args:
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save results
        model_name: Name of the model
        is_classification: Whether the model is a classification model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save metrics to JSON
    metrics_file = os.path.join(output_dir, f"{model_name}_metrics.json")

    # Remove non-serializable objects
    serializable_metrics = {}
    for key, value in metrics.items():
        if key not in ["confusion_matrix", "classification_report"]:
            serializable_metrics[key] = value

    with open(metrics_file, "w") as f:
        json.dump(serializable_metrics, f, indent=4)

    logger.info(f"Metrics saved to {metrics_file}")

    # For classification models, save confusion matrix
    if is_classification and "confusion_matrix" in metrics:
        cm_file = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            metrics["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=list(metrics["classification_report"].keys())[:-3],
            yticklabels=list(metrics["classification_report"].keys())[:-3]
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        plt.savefig(cm_file)
        plt.close()

        logger.info(f"Confusion matrix saved to {cm_file}")

def monitor_model_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Monitor model drift by comparing distributions of reference and current data

    Args:
        reference_data: Reference data (e.g., training data)
        current_data: Current data to check for drift
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        threshold: Threshold for drift detection

    Returns:
        Dict[str, Any]: Dictionary of drift metrics
    """
    logger.info("Monitoring model drift")

    drift_metrics = {
        "numerical_drift": {},
        "categorical_drift": {},
        "overall_drift_detected": False
    }

    # Check numerical features drift using Kolmogorov-Smirnov test
    from scipy.stats import ks_2samp

    for feature in numerical_features:
        if feature in reference_data.columns and feature in current_data.columns:
            ks_stat, p_value = ks_2samp(
                reference_data[feature].dropna(),
                current_data[feature].dropna()
            )

            drift_detected = p_value < threshold

            drift_metrics["numerical_drift"][feature] = {
                "ks_statistic": float(ks_stat),
                "p_value": float(p_value),
                "drift_detected": drift_detected
            }

            if drift_detected:
                drift_metrics["overall_drift_detected"] = True

    # Check categorical features drift using chi-square test
    from scipy.stats import chi2_contingency

    for feature in categorical_features:
        if feature in reference_data.columns and feature in current_data.columns:
            # Get value counts
            ref_counts = reference_data[feature].value_counts().to_dict()
            curr_counts = current_data[feature].value_counts().to_dict()

            # Get all unique values
            all_values = set(list(ref_counts.keys()) + list(curr_counts.keys()))

            # Create contingency table
            table = []
            for value in all_values:
                ref_count = ref_counts.get(value, 0)
                curr_count = curr_counts.get(value, 0)
                table.append([ref_count, curr_count])

            # Perform chi-square test
            try:
                chi2, p_value, _, _ = chi2_contingency(table)

                drift_detected = p_value < threshold

                drift_metrics["categorical_drift"][feature] = {
                    "chi2_statistic": float(chi2),
                    "p_value": float(p_value),
                    "drift_detected": drift_detected
                }

                if drift_detected:
                    drift_metrics["overall_drift_detected"] = True
            except Exception as e:
                logger.warning(f"Could not perform chi-square test for {feature}: {str(e)}")

    logger.info(f"Drift detection completed. Overall drift detected: {drift_metrics['overall_drift_detected']}")

    return drift_metrics