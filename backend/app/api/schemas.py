"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request schema for prediction endpoint."""
    symbol: Optional[str] = Field(None, description="Stock symbol (optional)")
    features: Optional[List[float]] = Field(None, description="Feature vector")
    return_components: bool = Field(False, description="Return individual model predictions")
    return_confidence: bool = Field(True, description="Return confidence intervals")


class PredictionResponse(BaseModel):
    """Response schema for prediction endpoint."""
    prediction: float = Field(..., description="Predicted price")
    model_version: str = Field(..., description="Model version used")
    components: Optional[Dict[str, Any]] = Field(None, description="Individual model predictions")
    confidence: Optional[Dict[str, float]] = Field(None, description="Confidence intervals")


class BacktestRequest(BaseModel):
    """Request schema for backtest endpoint."""
    start_date: Optional[str] = Field(None, description="Start date (ISO format)")
    end_date: Optional[str] = Field(None, description="End date (ISO format)")
    train_size: Optional[float] = Field(0.8, description="Training set proportion")


class BacktestResponse(BaseModel):
    """Response schema for backtest endpoint."""
    metrics: Dict[str, float] = Field(..., description="Performance metrics")
    predictions: List[float] = Field(..., description="Predictions")
    actuals: List[float] = Field(..., description="Actual values")
    dates: List[str] = Field(..., description="Dates")
    equity_curve: List[float] = Field(..., description="Equity curve")
    drawdown: List[float] = Field(..., description="Drawdown series")
    model_version: str = Field(..., description="Model version used")


class ExplainRequest(BaseModel):
    """Request schema for explain endpoint."""
    features: List[float] = Field(..., description="Feature vector to explain")


class ExplainResponse(BaseModel):
    """Response schema for explain endpoint."""
    shap_values: List[float] = Field(..., description="SHAP values")
    feature_names: List[str] = Field(..., description="Feature names")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance scores")
    base_value: Optional[float] = Field(None, description="Base prediction value")


class ModelInfoResponse(BaseModel):
    """Response schema for models endpoint."""
    version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model type")
    registered_at: str = Field(..., description="Registration timestamp")
    metadata: Dict[str, Any] = Field(..., description="Model metadata")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: bool = Field(..., description="Whether models are loaded")


class ActiveModelInfoResponse(BaseModel):
    """Response schema for active model info endpoint."""
    version: str = Field(..., description="Active model version")
    feature_names: List[str] = Field(..., description="Expected feature names")
    num_features: int = Field(..., description="Expected number of features")
    metadata: Dict[str, Any] = Field(..., description="Model metadata")


class TrainingRequest(BaseModel):
    """Request schema for training endpoint."""
    data: List[Dict[str, Any]] = Field(..., description="Training data as list of dictionaries")
    # Note: data will be converted to DataFrame on the server side


class TrainingResponse(BaseModel):
    """Response schema for training endpoint."""
    version: str = Field(..., description="New model version")
    feature_names: List[str] = Field(..., description="Feature names used")
    num_features: int = Field(..., description="Number of features")
    train_size: int = Field(..., description="Training set size")
    val_size: int = Field(..., description="Validation set size")
    test_size: int = Field(..., description="Test set size")
    results: Dict[str, Dict[str, float]] = Field(..., description="Training results for each model")
