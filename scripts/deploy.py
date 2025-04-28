#!/usr/bin/env python3
"""
Model Deployment Script

This script deploys a model to a Kubernetes cluster.
"""

import argparse
import logging
import os
import sys
import yaml
import subprocess
import time
from datetime import datetime

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.core.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Deploy a model to Kubernetes")

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to deploy"
    )

    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Version of the model to deploy (default: latest production version)"
    )

    parser.add_argument(
        "--namespace",
        type=str,
        default="default",
        help="Kubernetes namespace to deploy to"
    )

    parser.add_argument(
        "--replicas",
        type=int,
        default=2,
        help="Number of replicas to deploy"
    )

    parser.add_argument(
        "--cpu-request",
        type=str,
        default="100m",
        help="CPU request for each pod"
    )

    parser.add_argument(
        "--memory-request",
        type=str,
        default="512Mi",
        help="Memory request for each pod"
    )

    parser.add_argument(
        "--cpu-limit",
        type=str,
        default="500m",
        help="CPU limit for each pod"
    )

    parser.add_argument(
        "--memory-limit",
        type=str,
        default="1Gi",
        help="Memory limit for each pod"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to expose the service on"
    )

    parser.add_argument(
        "--enable-hpa",
        action="store_true",
        help="Enable Horizontal Pod Autoscaler"
    )

    parser.add_argument(
        "--min-replicas",
        type=int,
        default=2,
        help="Minimum number of replicas for HPA"
    )

    parser.add_argument(
        "--max-replicas",
        type=int,
        default=5,
        help="Maximum number of replicas for HPA"
    )

    parser.add_argument(
        "--target-cpu-utilization",
        type=int,
        default=80,
        help="Target CPU utilization percentage for HPA"
    )

    parser.add_argument(
        "--enable-monitoring",
        action="store_true",
        help="Enable Prometheus and Grafana monitoring"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Kubernetes manifests without applying them"
    )

    return parser.parse_args()

def generate_deployment_yaml(args):
    """Generate Kubernetes deployment YAML"""
    logger.info("Generating deployment YAML")

    # Set environment variables
    env_vars = [
        {"name": "MODEL_NAME", "value": args.model_name},
        {"name": "MODEL_VERSION", "value": args.model_version or "latest"},
        {"name": "MLFLOW_TRACKING_URI", "value": settings.MLFLOW_TRACKING_URI},
        {"name": "ENABLE_METRICS", "value": "true"},
        {"name": "PROMETHEUS_MULTIPROC_DIR", "value": "/tmp"},
    ]

    # Add AWS credentials if available
    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        env_vars.extend([
            {"name": "AWS_ACCESS_KEY_ID", "value": settings.AWS_ACCESS_KEY_ID},
            {"name": "AWS_SECRET_ACCESS_KEY", "value": settings.AWS_SECRET_ACCESS_KEY},
            {"name": "AWS_REGION", "value": settings.AWS_REGION},
            {"name": "S3_BUCKET", "value": settings.S3_BUCKET},
        ])

    # Create deployment YAML
    deployment = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": f"{args.model_name}-deployment",
            "namespace": args.namespace,
            "labels": {
                "app": args.model_name,
                "version": args.model_version or "latest"
            }
        },
        "spec": {
            "replicas": args.replicas,
            "selector": {
                "matchLabels": {
                    "app": args.model_name
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": args.model_name,
                        "version": args.model_version or "latest"
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "name": args.model_name,
                            "image": "model-api:latest",  # This should be your model API image
                            "ports": [
                                {"containerPort": args.port}
                            ],
                            "resources": {
                                "requests": {
                                    "cpu": args.cpu_request,
                                    "memory": args.memory_request
                                },
                                "limits": {
                                    "cpu": args.cpu_limit,
                                    "memory": args.memory_limit
                                }
                            },
                            "env": env_vars,
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": args.port
                                },
                                "initialDelaySeconds": 10,
                                "periodSeconds": 5
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": args.port
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            }
                        }
                    ]
                }
            }
        }
    }

    return deployment

def generate_service_yaml(args):
    """Generate Kubernetes service YAML"""
    logger.info("Generating service YAML")

    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": f"{args.model_name}-service",
            "namespace": args.namespace,
            "labels": {
                "app": args.model_name
            }
        },
        "spec": {
            "selector": {
                "app": args.model_name
            },
            "ports": [
                {
                    "port": args.port,
                    "targetPort": args.port,
                    "protocol": "TCP"
                }
            ],
            "type": "ClusterIP"
        }
    }

    return service

def generate_hpa_yaml(args):
    """Generate Kubernetes HPA YAML"""
    logger.info("Generating HPA YAML")

    hpa = {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {
            "name": f"{args.model_name}-hpa",
            "namespace": args.namespace
        },
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": f"{args.model_name}-deployment"
            },
            "minReplicas": args.min_replicas,
            "maxReplicas": args.max_replicas,
            "metrics": [
                {
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": args.target_cpu_utilization
                        }
                    }
                }
            ]
        }
    }

    return hpa

def apply_kubernetes_manifest(manifest, dry_run=False):
    """Apply a Kubernetes manifest"""
    # Write manifest to temporary file
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"/tmp/manifest_{timestamp}.yaml"

    with open(filename, "w") as f:
        yaml.dump(manifest, f)

    logger.info(f"Manifest written to {filename}")

    if dry_run:
        logger.info("Dry run mode, not applying manifest")
        with open(filename, "r") as f:
            manifest_str = f.read()
        logger.info(f"Manifest content:\n{manifest_str}")
        return True

    # Apply manifest
    try:
        logger.info(f"Applying manifest {filename}")
        result = subprocess.run(
            ["kubectl", "apply", "-f", filename],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Manifest applied successfully: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error applying manifest: {e.stderr}")
        return False
    finally:
        # Clean up temporary file
        os.remove(filename)

def deploy_monitoring(args):
    """Deploy Prometheus and Grafana for monitoring"""
    logger.info("Deploying monitoring stack")

    # Check if monitoring manifests exist
    prometheus_path = "k8s/monitoring/prometheus.yaml"
    grafana_path = "k8s/monitoring/grafana.yaml"

    if not os.path.exists(prometheus_path) or not os.path.exists(grafana_path):
        logger.error("Monitoring manifests not found")
        return False

    # Apply monitoring manifests
    try:
        if args.dry_run:
            logger.info("Dry run mode, not applying monitoring manifests")
            with open(prometheus_path, "r") as f:
                prometheus_yaml = f.read()
            with open(grafana_path, "r") as f:
                grafana_yaml = f.read()
            logger.info(f"Prometheus manifest:\n{prometheus_yaml}")
            logger.info(f"Grafana manifest:\n{grafana_yaml}")
            return True

        logger.info("Applying Prometheus manifest")
        prometheus_result = subprocess.run(
            ["kubectl", "apply", "-f", prometheus_path],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Prometheus manifest applied successfully: {prometheus_result.stdout}")

        logger.info("Applying Grafana manifest")
        grafana_result = subprocess.run(
            ["kubectl", "apply", "-f", grafana_path],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Grafana manifest applied successfully: {grafana_result.stdout}")

        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error deploying monitoring: {e.stderr}")
        return False

def wait_for_deployment(args):
    """Wait for deployment to be ready"""
    if args.dry_run:
        logger.info("Dry run mode, not waiting for deployment")
        return True

    logger.info(f"Waiting for deployment {args.model_name}-deployment to be ready")

    max_retries = 30
    retry_interval = 5

    for i in range(max_retries):
        try:
            result = subprocess.run(
                [
                    "kubectl", "rollout", "status", "deployment",
                    f"{args.model_name}-deployment",
                    "-n", args.namespace
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            logger.info(f"Deployment ready: {result.stdout}")
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Deployment not ready yet, retrying in {retry_interval} seconds...")
            time.sleep(retry_interval)

    logger.error(f"Deployment not ready after {max_retries * retry_interval} seconds")
    return False

def deploy_model(args):
    """Deploy a model to Kubernetes"""
    logger.info(f"Deploying model {args.model_name} version {args.model_version or 'latest'} to Kubernetes")

    # Generate Kubernetes manifests
    deployment = generate_deployment_yaml(args)
    service = generate_service_yaml(args)

    # Apply deployment and service
    deployment_success = apply_kubernetes_manifest(deployment, args.dry_run)
    service_success = apply_kubernetes_manifest(service, args.dry_run)

    if not deployment_success or not service_success:
        logger.error("Failed to deploy model")
        return False

    # Apply HPA if enabled
    if args.enable_hpa:
        hpa = generate_hpa_yaml(args)
        hpa_success = apply_kubernetes_manifest(hpa, args.dry_run)

        if not hpa_success:
            logger.error("Failed to deploy HPA")
            return False

    # Deploy monitoring if enabled
    if args.enable_monitoring:
        monitoring_success = deploy_monitoring(args)

        if not monitoring_success:
            logger.error("Failed to deploy monitoring")
            return False

    # Wait for deployment to be ready
    if not args.dry_run:
        ready = wait_for_deployment(args)

        if not ready:
            logger.error("Deployment not ready")
            return False

    logger.info(f"Model {args.model_name} version {args.model_version or 'latest'} deployed successfully")
    return True

if __name__ == "__main__":
    args = parse_args()
    success = deploy_model(args)
    sys.exit(0 if success else 1)