from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # API settings
    API_V1_STR: str = "/v1"
    PROJECT_NAME: str = "ML Model API"

    # MLflow settings
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "default")

    # Model settings
    MODEL_NAME: str = os.getenv("MODEL_NAME", "ml-model")
    MODEL_STAGE: str = os.getenv("MODEL_STAGE", "Production")

    # AWS settings for MLflow artifact storage
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "mlflow-artifacts")

    # Monitoring settings
    PROMETHEUS_MULTIPROC_DIR: str = os.getenv("PROMETHEUS_MULTIPROC_DIR", "/tmp")
    ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "True").lower() in ("true", "1", "t")

    # Optimization settings
    QUANTIZE_MODEL: bool = os.getenv("QUANTIZE_MODEL", "False").lower() in ("true", "1", "t")
    USE_ONNX: bool = os.getenv("USE_ONNX", "False").lower() in ("true", "1", "t")

    class Config:
        case_sensitive = True

# Create global settings object
settings = Settings()