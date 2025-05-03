# Optimized ML Model Deployment Pipeline

An end-to-end MLOps pipeline for efficient model training, deployment, and monitoring.

## Features

- **Automated CI/CD**: GitHub Actions workflows for model training and deployment
- **Model Versioning**: MLflow for experiment tracking and model registry
- **Containerized Deployments**: Docker and Kubernetes for scalable deployments
- **Monitoring System**: Prometheus and Grafana for model performance metrics
- **Model Optimization**: ONNX and TensorFlow Lite for model quantization and optimization

## Architecture

The pipeline consists of the following components:

1. **Model Training**: Train ML models with scikit-learn and track experiments with MLflow
2. **Model Registry**: Version and store models in MLflow's model registry
3. **Model Optimization**: Optimize models using ONNX and TensorFlow Lite
4. **Model Deployment**: Deploy models as REST APIs using FastAPI
5. **Containerization**: Package the application using Docker
6. **Orchestration**: Deploy and scale using Kubernetes
7. **Monitoring**: Track model performance with Prometheus and Grafana

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Kubernetes cluster (for production deployment)
- MLflow server (or use the included Docker Compose setup)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Optimized-ML-Model-Deployment-Pipeline.git
   cd Optimized-ML-Model-Deployment-Pipeline
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the local development environment:
   ```bash
   docker-compose up -d
   ```

### Training a Model

Use the training script to train a new model:

```bash
python scripts/train.py \
  --data-path data/your_dataset.csv \
  --model-type classification \
  --algorithm random_forest \
  --target-column target \
  --numerical-features "feature1,feature2,feature3" \
  --register-model \
  --model-name "your-model-name"
```

### Optimizing a Model

Optimize a trained model for inference:

```bash
python scripts/optimize_model.py \
  --model-uri "models:/your-model-name/Production" \
  --convert-to-onnx \
  --quantize \
  --register-model \
  --model-name "your-model-name-optimized"
```

### Deploying a Model

Deploy a model to Kubernetes:

```bash
python scripts/deploy.py \
  --model-name "your-model-name" \
  --enable-hpa \
  --enable-monitoring
```

## API Endpoints

The deployed model exposes the following API endpoints:

- `POST /api/predict`: Make a single prediction
- `POST /api/batch-predict`: Make batch predictions
- `GET /api/model-info`: Get information about the deployed model
- `GET /health`: Health check endpoint
- `GET /metrics`: Prometheus metrics endpoint

## Monitoring

The pipeline includes a comprehensive monitoring setup:

- **Prometheus**: Collects metrics from the model API
- **Grafana**: Visualizes metrics with pre-configured dashboards
- **Model Drift Detection**: Monitors for data and concept drift

Access Grafana at `http://localhost:3000` (local) or through your Kubernetes ingress.

## CI/CD Pipeline

The GitHub Actions workflow automates:

1. Running tests
2. Training models
3. Optimizing models
4. Building and pushing Docker images
5. Deploying to Kubernetes

## Directory Structure

```
.
├── app/                    # Application code
│   ├── api/                # API endpoints
│   ├── core/               # Core functionality
│   └── ml/                 # ML model code
├── k8s/                    # Kubernetes manifests
│   └── monitoring/         # Monitoring setup
├── scripts/                # Utility scripts
├── tests/                  # Test suite
├── .github/workflows/      # CI/CD workflows
├── docker-compose.yml      # Local development setup
├── Dockerfile              # Container definition
└── requirements.txt        # Python dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License
