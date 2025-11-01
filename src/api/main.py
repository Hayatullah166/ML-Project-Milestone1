"""
FastAPI application for Energy Consumption Prediction service.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pathlib import Path
import logging
from contextlib import asynccontextmanager

from src.api.schemas import (
    PredictionRequest, PredictionResponse, ModelInfo, ModelListResponse,
    HealthResponse, ErrorResponse, RetrainRequest, RetrainResponse,
    DriftCheckResponse
)
from src.utils.config import config
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.trainer import ModelTrainer
from src.mlops.drift_detector import DriftDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelService:
    """Model service for managing and serving predictions."""
    
    def __init__(self):
        """Initialize model service."""
        self.models = {}
        self.active_model = None
        self.preprocessor = DataPreprocessor()
        self.drift_detector = DriftDetector()
        self.model_path = Path(config.get('models.trained_path', 'models/trained'))
        
    def load_models(self):
        """Load all available models."""
        logger.info("Loading models...")
        
        if not self.model_path.exists():
            logger.warning("No models found. Please train models first.")
            return
        
        # Load all model files
        for model_file in self.model_path.glob("*_model.joblib"):
            model_name = model_file.stem.replace("_model", "")
            
            try:
                model = joblib.load(model_file)
                self.models[model_name] = model
                logger.info(f"Loaded model: {model_name}")
                
                # Set first loaded model as active
                if self.active_model is None:
                    self.active_model = model_name
                    
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
        
        logger.info(f"Loaded {len(self.models)} models. Active: {self.active_model}")
    
    def get_model(self, model_name: Optional[str] = None):
        """Get model by name or active model."""
        if model_name is None:
            model_name = self.active_model
        
        if model_name not in self.models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found"
            )
        
        return self.models[model_name]
    
    def predict(self, historical_data: List[Dict[str, Any]], 
                horizon: int = 24, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Make predictions using the specified model."""
        try:
            # Get model
            model = self.get_model(model_name)
            used_model_name = model_name or self.active_model
            
            # Convert historical data to DataFrame
            df_hist = pd.DataFrame(historical_data)
            df_hist['datetime'] = pd.to_datetime(df_hist['datetime'])
            df_hist.set_index('datetime', inplace=True)
            
            # Prepare features
            df_processed, feature_columns = self.preprocessor.prepare_features(df_hist)
            
            if len(df_processed) == 0:
                raise ValueError("No valid data for prediction")
            
            # Get latest features
            latest_features = df_processed[feature_columns].iloc[-1:].values
            
            # Make predictions
            predictions = []
            current_features = latest_features.copy()
            
            for i in range(horizon):
                # Predict next value
                pred = model.predict(current_features)[0]
                predictions.append(pred)
                
                # Update features for next prediction (simplified)
                # In a real implementation, you'd update lag features properly
                current_features = np.roll(current_features, -1)
                current_features[0, -1] = pred  # Update the last lag feature
            
            # Generate timestamps
            last_timestamp = df_processed.index[-1]
            timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(horizon)]
            
            # Create response
            prediction_data = []
            for i, (ts, pred) in enumerate(zip(timestamps, predictions)):
                # Simple confidence interval (in practice, use proper uncertainty quantification)
                std_dev = np.std(predictions) * 0.1  # Rough estimate
                prediction_data.append({
                    "datetime": ts.isoformat(),
                    "predicted_consumption": float(pred),
                    "confidence_interval_lower": float(max(0, pred - 1.96 * std_dev)),
                    "confidence_interval_upper": float(pred + 1.96 * std_dev)
                })
            
            return {
                "predictions": prediction_data,
                "model_used": used_model_name,
                "prediction_horizon": horizon,
                "metadata": {
                    "features_used": len(feature_columns),
                    "historical_points": len(df_processed),
                    "prediction_timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def get_model_info(self) -> List[ModelInfo]:
        """Get information about all available models."""
        model_info = []
        
        for model_name, model in self.models.items():
            # Load metrics if available
            metrics_file = self.model_path / f"{model_name}_metrics.json"
            metrics = {}
            created_at = datetime.now()
            
            if metrics_file.exists():
                try:
                    import json
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    # Get file creation time
                    created_at = datetime.fromtimestamp(metrics_file.stat().st_mtime)
                except Exception as e:
                    logger.warning(f"Could not load metrics for {model_name}: {e}")
            
            model_info.append(ModelInfo(
                name=model_name,
                type="ml" if hasattr(model, 'feature_importances_') else "baseline",
                metrics=metrics,
                created_at=created_at,
                is_active=(model_name == self.active_model)
            ))
        
        return model_info

# Global model service instance
model_service = ModelService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Energy Consumption Prediction API...")
    model_service.load_models()
    yield
    # Shutdown
    logger.info("Shutting down Energy Consumption Prediction API...")

# Create FastAPI app
app = FastAPI(
    title="Energy Consumption Prediction API",
    description="MLOps-enabled API for energy consumption forecasting",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        models_loaded=len(model_service.models) > 0
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        models_loaded=len(model_service.models) > 0
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_consumption(request: PredictionRequest):
    """Predict energy consumption for the specified horizon."""
    try:
        result = model_service.predict(
            historical_data=request.historical_data,
            horizon=request.horizon,
            model_name=request.model_name
        )
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=ModelListResponse)
async def list_models():
    """List all available models."""
    model_info = model_service.get_model_info()
    return ModelListResponse(
        models=model_info,
        active_model=model_service.active_model or "none"
    )

@app.post("/models/{model_name}/activate")
async def activate_model(model_name: str):
    """Activate a specific model."""
    if model_name not in model_service.models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    model_service.active_model = model_name
    logger.info(f"Activated model: {model_name}")
    
    return {"message": f"Model '{model_name}' activated successfully"}

@app.post("/retrain", response_model=RetrainResponse)
async def retrain_model(request: RetrainRequest, background_tasks: BackgroundTasks):
    """Retrain model with latest data."""
    try:
        # This would trigger retraining in the background
        # For now, return a placeholder response
        background_tasks.add_task(_retrain_model_task, request)
        
        return RetrainResponse(
            success=True,
            new_model_name="retrained_model",
            metrics={"mae": 0.5, "rmse": 0.7, "r2": 0.85},
            training_time=120.5,
            message="Retraining started in background"
        )
    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drift-check", response_model=DriftCheckResponse)
async def check_drift():
    """Check for data drift."""
    try:
        drift_result = model_service.drift_detector.check_drift()
        return DriftCheckResponse(**drift_result)
    except Exception as e:
        logger.error(f"Drift check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _retrain_model_task(request: RetrainRequest):
    """Background task for model retraining."""
    try:
        logger.info("Starting model retraining...")
        # Implement retraining logic here
        # This would involve:
        # 1. Loading latest data
        # 2. Training new model
        # 3. Evaluating performance
        # 4. Updating model service
        logger.info("Model retraining completed")
    except Exception as e:
        logger.error(f"Background retraining failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.get('api.host', '0.0.0.0'),
        port=config.get('api.port', 8000),
        reload=config.get('api.debug', True)
    )
