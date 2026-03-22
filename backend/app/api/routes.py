"""
FastAPI route handlers for stock prediction API.
"""
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, List, Dict, Any
import logging

from app.api.schemas import (
    PredictionRequest, PredictionResponse,
    BacktestRequest, BacktestResponse,
    ExplainRequest, ExplainResponse,
    ModelInfoResponse, HealthResponse, ActiveModelInfoResponse,
    TrainingRequest, TrainingResponse
)
from app.services.model_registry import ModelRegistry
from app.services.prediction_service import PredictionService
from app.services.backtesting import BacktestingEngine
from app.services.explainability import ExplainabilityService
from app.services.feature_engineering import FeatureEngineer
from app.services.training_service import TrainingService
from app.utils.data_loader import DataLoader
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix=settings.API_PREFIX, tags=["predictions"])


# Global service instances (singleton pattern)
_model_registry: Optional[ModelRegistry] = None
_feature_engineer: Optional[FeatureEngineer] = None
_prediction_service: Optional[PredictionService] = None
_explainability_service: Optional[ExplainabilityService] = None


def get_model_registry() -> ModelRegistry:
    """Get model registry instance (singleton)."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
        # Try to load models on first access
        try:
            _model_registry.load_models()
            # Verify at least one model was loaded
            if not any([
                _model_registry.lgb_model is not None,
                _model_registry.xgb_model is not None,
                _model_registry.lstm_model is not None,
                _model_registry.nn_model is not None
            ]):
                logger.warning("Model registry initialized but no models were loaded")
        except Exception as e:
            logger.error(f"Failed to load models: {e}", exc_info=True)
            logger.warning("Model registry initialized but models not loaded. Will attempt to load on first request.")
    return _model_registry


def get_feature_engineer() -> FeatureEngineer:
    """Get feature engineer instance (singleton)."""
    global _feature_engineer
    if _feature_engineer is None:
        _feature_engineer = FeatureEngineer(feature_version=settings.FEATURE_VERSION)
    return _feature_engineer


def get_prediction_service(
    registry: ModelRegistry = Depends(get_model_registry),
    feature_engineer: FeatureEngineer = Depends(get_feature_engineer)
) -> PredictionService:
    """Get prediction service instance (singleton)."""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService(
            model_registry=registry,
            feature_engineer=feature_engineer,
            lgb_weight=0.4,  # LightGBM weight
            xgb_weight=0.4,  # XGBoost weight
            lstm_weight=0.2,  # LSTM weight
            nn_weight=0.0  # Legacy NN weight
        )
    return _prediction_service


def get_explainability_service(
    registry: ModelRegistry = Depends(get_model_registry)
) -> ExplainabilityService:
    """Get explainability service instance (singleton)."""
    global _explainability_service
    if _explainability_service is None:
        _explainability_service = ExplainabilityService(model_registry=registry)
    return _explainability_service


@router.get("/health", response_model=HealthResponse)
async def health_check(
    registry: ModelRegistry = Depends(get_model_registry)
) -> HealthResponse:
    """Health check endpoint."""
    # Check for all available model types
    models_loaded = (
        registry.lgb_model is not None or
        registry.xgb_model is not None or
        registry.lstm_model is not None or
        registry.nn_model is not None
    )
    
    return HealthResponse(
        status="healthy" if models_loaded else "models_not_loaded",
        version=settings.API_VERSION,
        models_loaded=models_loaded
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service)
) -> PredictionResponse:
    """
    Get stock price prediction.
    
    Accepts either:
    - Feature vector directly
    - Symbol (will load latest data and engineer features)
    """
    try:
        # If features provided directly, use them
        if request.features:
            features = np.array(request.features).reshape(1, -1)
        elif request.symbol:
            # Load data and engineer features
            # For now, raise error - implement data loading if needed
            raise HTTPException(
                status_code=501,
                detail="Symbol-based prediction not yet implemented. Provide features directly."
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'features' or 'symbol' must be provided"
            )
        
        # Make prediction
        result = prediction_service.predict(
            features,
            return_components=request.return_components,
            return_confidence=request.return_confidence
        )
        
        return PredictionResponse(**result)
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest", response_model=BacktestResponse)
async def backtest(
    request: BacktestRequest,
    registry: ModelRegistry = Depends(get_model_registry),
    feature_engineer: FeatureEngineer = Depends(get_feature_engineer),
    prediction_service: PredictionService = Depends(get_prediction_service)
) -> BacktestResponse:
    """
    Run walk-forward backtest.
    
    Uses data from configured data directory.
    """
    try:
        # Load data
        data_loader = DataLoader(settings.DATA_DIR / "FINAL_FEATURES_OUT.csv")
        df = data_loader.load_raw_data()
        
        # Check if data already has features or needs feature engineering
        # FINAL_FEATURES_OUT.csv already has features, so check for Close Price
        if "Close Price" not in df.columns and "Target_Close" not in df.columns:
            # Need to engineer features from raw data
            df = feature_engineer.create_features(df)
            df["Target_Close"] = df["Close Price"].shift(-1)
            df = df.dropna(subset=["Target_Close"])
        elif "Target_Close" not in df.columns and "Close Price" in df.columns:
            # Has Close Price but not Target_Close, add it
            df["Target_Close"] = df["Close Price"].shift(-1)
            df = df.dropna(subset=["Target_Close"])
        
        # Filter by date if provided (only works with DatetimeIndex)
        if request.start_date and isinstance(df.index, pd.DatetimeIndex):
            start_dt = pd.to_datetime(request.start_date)
            df = df[df.index >= start_dt]
        if request.end_date and isinstance(df.index, pd.DatetimeIndex):
            end_dt = pd.to_datetime(request.end_date)
            df = df[df.index <= end_dt]
        
        # Run backtest
        backtest_engine = BacktestingEngine(
            model_registry=registry,
            prediction_service=prediction_service,
            feature_engineer=feature_engineer,
            train_size=request.train_size or 0.8
        )
        
        results = backtest_engine.run_backtest(df, target_col="Target_Close")
        
        return BacktestResponse(**results)
    
    except Exception as e:
        logger.error(f"Backtest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain", response_model=ExplainResponse)
async def explain(
    request: ExplainRequest,
    explain_service: ExplainabilityService = Depends(get_explainability_service),
    feature_engineer: FeatureEngineer = Depends(get_feature_engineer),
    registry: ModelRegistry = Depends(get_model_registry)
) -> ExplainResponse:
    """
    Get SHAP-based explanation for a prediction.
    """
    try:
        features = np.array(request.features).reshape(1, -1)
        
        # Get feature names from model metadata (most reliable)
        version = registry.registry.get("active_version") or registry.current_version
        feature_names = None
        if version:
            model_info = registry.get_model_info(version)
            if model_info and "xgboost" in model_info:
                xgb_meta = model_info["xgboost"].get("metadata", {})
                feature_names = xgb_meta.get("feature_names", [])
        
        # Fallback to feature engineer's names
        if not feature_names:
            feature_names = feature_engineer.get_feature_names()
        
        # Initialize explainer if needed
        if explain_service.explainer is None:
            if not feature_names:
                raise HTTPException(
                    status_code=400,
                    detail="Feature names not available. Cannot initialize explainer without model metadata."
                )
            
            # Use sample data for initialization
            data_loader = DataLoader(settings.DATA_DIR / "FINAL_FEATURES_OUT.csv")
            df = data_loader.load_raw_data()
            
            # Validate that all required features exist in the data
            missing_features = [f for f in feature_names if f not in df.columns]
            if missing_features:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing features in data: {missing_features}"
                )
            
            # Use exactly the feature names from model metadata (in correct order)
            feature_cols = feature_names
            X_sample = df[feature_cols].values[:100]  # Sample for background
            
            # Validate feature count
            if X_sample.shape[1] != len(feature_names):
                raise HTTPException(
                    status_code=500,
                    detail=f"Feature count mismatch: expected {len(feature_names)}, got {X_sample.shape[1]}"
                )
            
            explain_service.initialize_explainer(X_sample, feature_cols)
        
        # Validate feature count matches model expectations
        if features.shape[1] != len(feature_names):
            raise HTTPException(
                status_code=400,
                detail=f"Feature count mismatch: model expects {len(feature_names)} features, got {features.shape[1]}"
            )
        
        # Get explanation
        explanation = explain_service.explain_prediction(features, feature_names)
        
        return ExplainResponse(**explanation)
    
    except Exception as e:
        logger.error(f"Explanation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=List[ModelInfoResponse])
async def list_models(
    registry: ModelRegistry = Depends(get_model_registry)
) -> List[ModelInfoResponse]:
    """List all registered model versions."""
    try:
        versions = registry.list_versions()
        models = []
        
        for version in versions:
            info = registry.get_model_info(version)
            for model_type, model_data in info.items():
                models.append(ModelInfoResponse(
                    version=version,
                    model_type=model_type,
                    registered_at=model_data.get("registered_at", ""),
                    metadata=model_data.get("metadata", {})
                ))
        
        return models
    
    except Exception as e:
        logger.error(f"Failed to list models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/info", response_model=ActiveModelInfoResponse)
async def get_active_model_info(
    registry: ModelRegistry = Depends(get_model_registry)
) -> ActiveModelInfoResponse:
    """Get information about the active model including feature requirements."""
    try:
        version = registry.registry.get("active_version") or registry.current_version
        if version is None:
            raise HTTPException(status_code=404, detail="No active model version found")
        
        model_info = registry.get_model_info(version)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model version {version} not found")
        
        # Get feature names from xgboost metadata (primary model)
        feature_names = []
        if "xgboost" in model_info:
            xgb_meta = model_info["xgboost"].get("metadata", {})
            feature_names = xgb_meta.get("feature_names", [])
        elif "neural_network" in model_info:
            nn_meta = model_info["neural_network"].get("metadata", {})
            input_dim = nn_meta.get("input_dim", 0)
            feature_names = [f"Feature_{i+1}" for i in range(input_dim)]
        
        num_features = len(feature_names) if feature_names else 0
        
        # Combine metadata from both models
        combined_metadata = {}
        if "xgboost" in model_info:
            combined_metadata["xgboost"] = model_info["xgboost"].get("metadata", {})
        if "neural_network" in model_info:
            combined_metadata["neural_network"] = model_info["neural_network"].get("metadata", {})
        
        return ActiveModelInfoResponse(
            version=version,
            feature_names=feature_names,
            num_features=num_features,
            metadata=combined_metadata
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/status")
async def get_model_status(
    registry: ModelRegistry = Depends(get_model_registry)
) -> Dict[str, Any]:
    """Get detailed diagnostic information about model loading status."""
    try:
        status = registry.get_loading_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get model status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train", response_model=TrainingResponse)
async def train_models(
    request: TrainingRequest,
    registry: ModelRegistry = Depends(get_model_registry)
) -> TrainingResponse:
    """
    Train models with new data.
    
    The data should contain stock price data with required columns:
    - Open Price, High Price, Low Price, Close Price
    - Total Traded Quantity, Turnover (optional)
    
    After training, the new model will be automatically loaded and used for predictions.
    """
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Validate required columns
        required_cols = ["Open Price", "High Price", "Low Price", "Close Price"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}. Data must contain: {required_cols}"
            )
        
        if len(df) < 100:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: need at least 100 rows, got {len(df)}"
            )
        
        # Initialize training service
        training_service = TrainingService(registry)
        
        # Train models
        results = training_service.train_models(df)
        
        return TrainingResponse(
            version=results["version"],
            feature_names=results["feature_names"],
            num_features=results["num_features"],
            train_size=results["train_size"],
            val_size=results["val_size"],
            test_size=results["test_size"],
            results={
                "lightgbm": results["lightgbm"],
                "xgboost": results["xgboost"],
                "lstm": results["lstm"]
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

