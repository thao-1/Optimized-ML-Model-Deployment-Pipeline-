from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import mlflow
import os

from app.api.endpoints import router as api_router
from app.core.config import settings
from app.core.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ML Model API",
    description="API for ML model inference",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix="/api")

# Setup MLflow
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application")
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI: {settings.MLFLOW_TRACKING_URI}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the application")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)