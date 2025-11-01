"""
Pydantic schemas for FastAPI Energy Consumption Prediction service.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd

class PredictionRequest(BaseModel):
    """Request schema for energy consumption prediction."""
    
    # Historical data for prediction
    historical_data: List[Dict[str, Any]] = Field(
        ..., 
        description="List of historical energy consumption records",
        example=[
            {
                "datetime": "2023-01-01T00:00:00",
                "Global_active_power": 1.2,
                "Global_reactive_power": 0.3,
                "Voltage": 240.5,
                "Global_intensity": 5.1,
                "Sub_metering_1": 0.0,
                "Sub_metering_2": 0.0,
                "Sub_metering_3": 0.0
            }
        ]
    )
    
    # Prediction horizon (hours ahead)
    horizon: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Number of hours ahead to predict (1-168)"
    )
    
    # Optional model name to use
    model_name: Optional[str] = Field(
        default=None,
        description="Specific model to use for prediction"
    )

class PredictionResponse(BaseModel):
    """Response schema for energy consumption prediction."""
    
    predictions: List[Dict[str, Any]] = Field(
        ...,
        description="List of predicted energy consumption values",
        example=[
            {
                "datetime": "2023-01-02T00:00:00",
                "predicted_consumption": 1.15,
                "confidence_interval_lower": 0.95,
                "confidence_interval_upper": 1.35
            }
        ]
    )
    
    model_used: str = Field(
        ...,
        description="Name of the model used for prediction"
    )
    
    prediction_horizon: int = Field(
        ...,
        description="Number of hours predicted ahead"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the prediction"
    )

class ModelInfo(BaseModel):
    """Schema for model information."""
    
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type (baseline/ml)")
    metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    created_at: datetime = Field(..., description="Model creation timestamp")
    is_active: bool = Field(..., description="Whether model is currently active")

class ModelListResponse(BaseModel):
    """Response schema for model list."""
    
    models: List[ModelInfo] = Field(..., description="List of available models")
    active_model: str = Field(..., description="Currently active model name")

class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    models_loaded: bool = Field(..., description="Whether models are loaded")

class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(..., description="Error timestamp")

class RetrainRequest(BaseModel):
    """Request schema for model retraining."""
    
    model_name: Optional[str] = Field(
        default=None,
        description="Specific model to retrain (if None, retrains best model)"
    )
    
    use_latest_data: bool = Field(
        default=True,
        description="Whether to use latest available data"
    )
    
    validation_split: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Validation split ratio"
    )

class RetrainResponse(BaseModel):
    """Response schema for model retraining."""
    
    success: bool = Field(..., description="Whether retraining was successful")
    new_model_name: str = Field(..., description="Name of the newly trained model")
    metrics: Dict[str, float] = Field(..., description="New model performance metrics")
    training_time: float = Field(..., description="Training time in seconds")
    message: str = Field(..., description="Status message")

class DriftCheckResponse(BaseModel):
    """Response schema for drift detection check."""
    
    drift_detected: bool = Field(..., description="Whether drift was detected")
    drift_score: float = Field(..., description="Drift score (0-1)")
    drift_details: Dict[str, Any] = Field(..., description="Detailed drift information")
    recommendation: str = Field(..., description="Recommendation based on drift")
    timestamp: datetime = Field(..., description="Drift check timestamp")
