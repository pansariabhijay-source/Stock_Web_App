"""
Configuration management for the backend API.
Uses environment variables with sensible defaults.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    API_TITLE: str = "Stock Prediction API"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    
    # Model Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    MODELS_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"
    
    # Model Registry
    XGBOOST_MODEL_PATH: Optional[str] = None
    NN_MODEL_PATH: Optional[str] = None
    SCALER_PATH: Optional[str] = None
    
    # Feature Engineering
    FEATURE_VERSION: str = "v1"
    LOOKBACK_WINDOW: int = 30  # Days of historical data needed
    
    # Prediction Settings
    ENSEMBLE_XGB_WEIGHT: float = 0.7  # Default XGBoost weight
    ENSEMBLE_NN_WEIGHT: float = 0.3   # Default NN weight
    CONFIDENCE_ALPHA: float = 0.05    # 95% confidence interval
    
    # Backtesting
    BACKTEST_TRAIN_SIZE: float = 0.8
    BACKTEST_STEP_SIZE: int = 1  # Days to step forward
    
    # Caching (Redis)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    CACHE_TTL: int = 3600  # 1 hour
    
    # Monitoring
    ENABLE_METRICS: bool = True
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

