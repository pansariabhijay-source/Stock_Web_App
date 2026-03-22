"""
Production-grade prediction service for ensemble model predictions.
Supports LightGBM, XGBoost, LSTM with intelligent ensemble weighting.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging

from app.services.model_registry import ModelRegistry
from app.services.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Production-grade service for making ensemble predictions.
    
    Supports multiple models:
    - LightGBM (preferred for speed and accuracy)
    - XGBoost (robust baseline)
    - LSTM (time-series patterns)
    - Neural Network (legacy support)
    
    Uses intelligent ensemble weighting based on model performance.
    """
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        feature_engineer: FeatureEngineer,
        lgb_weight: float = 0.4,
        xgb_weight: float = 0.4,
        lstm_weight: float = 0.2,
        nn_weight: float = 0.0
    ):
        """
        Initialize prediction service.
        
        Args:
            model_registry: Model registry instance
            feature_engineer: Feature engineering instance
            lgb_weight: LightGBM weight (default: 0.4)
            xgb_weight: XGBoost weight (default: 0.4)
            lstm_weight: LSTM weight (default: 0.2)
            nn_weight: Neural Network weight (default: 0.0, legacy)
        """
        self.model_registry = model_registry
        self.feature_engineer = feature_engineer
        
        # Normalize weights
        total = lgb_weight + xgb_weight + lstm_weight + nn_weight
        if total > 0:
            self.lgb_weight = lgb_weight / total
            self.xgb_weight = xgb_weight / total
            self.lstm_weight = lstm_weight / total
            self.nn_weight = nn_weight / total
        else:
            # Default equal weights if all zero
            self.lgb_weight = 0.4
            self.xgb_weight = 0.4
            self.lstm_weight = 0.2
            self.nn_weight = 0.0
    
    def _get_available_models(self) -> Dict[str, bool]:
        """Check which models are available."""
        return {
            "lightgbm": self.model_registry.lgb_model is not None,
            "xgboost": self.model_registry.xgb_model is not None,
            "lstm": self.model_registry.lstm_model is not None,
            "neural_network": self.model_registry.nn_model is not None
        }
    
    def _validate_models_loaded(self) -> None:
        """Validate that at least one model is loaded."""
        available = self._get_available_models()
        if not any(available.values()):
            raise ValueError(
                "No models loaded. Call model_registry.load_models() first. "
                f"Available models: {available}"
            )
    
    def predict(
        self,
        features: np.ndarray,
        return_components: bool = False,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Make ensemble prediction using available models.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            return_components: Whether to return individual model predictions
            return_confidence: Whether to compute confidence intervals
            
        Returns:
            Dictionary with predictions and metadata
        """
        self._validate_models_loaded()
        
        # Ensure features is 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Validate feature dimensions
        expected_feature_count = self._get_expected_feature_count()
        if expected_feature_count and features.shape[1] != expected_feature_count:
            raise ValueError(
                f"Feature dimension mismatch: model expects {expected_feature_count} features, "
                f"but received {features.shape[1]}. Please provide exactly {expected_feature_count} features."
            )
        
        # Get feature names for XGBoost (it requires feature names)
        feature_names = self._get_feature_names()
        
        # Get predictions from all available models
        predictions = {}
        available = self._get_available_models()
        
        try:
            if available["lightgbm"]:
                lgb_pred = self.model_registry.lgb_model.predict(features)
                predictions["lightgbm"] = self._normalize_prediction(lgb_pred)
            
            if available["xgboost"]:
                # XGBoost requires feature names if model was trained with them
                xgb_pred = self.model_registry.xgb_model.predict(features, feature_names=feature_names)
                predictions["xgboost"] = self._normalize_prediction(xgb_pred)
            
            if available["lstm"]:
                lstm_pred = self.model_registry.lstm_model.predict(features)
                predictions["lstm"] = self._normalize_prediction(lstm_pred)
            
            if available["neural_network"]:
                nn_pred = self.model_registry.nn_model.predict(features)
                predictions["neural_network"] = self._normalize_prediction(nn_pred)
        
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            raise ValueError(f"Prediction failed: {str(e)}")
        
        if not predictions:
            raise ValueError("No predictions generated from any model")
        
        # Calculate ensemble prediction with dynamic weights
        ensemble_pred = self._ensemble_predictions(predictions, available)
        
        # Convert to Python types
        def to_python_value(arr):
            """Convert numpy array to Python float or list."""
            if arr.ndim == 0:
                return float(arr.item())
            elif len(arr) == 1:
                return float(arr[0])
            else:
                return arr.tolist()
        
        result = {
            "prediction": to_python_value(ensemble_pred),
            "model_version": self.model_registry.current_version,
            "models_used": list(predictions.keys())
        }
        
        if return_components:
            components = {k: to_python_value(v) for k, v in predictions.items()}
            components["ensemble"] = to_python_value(ensemble_pred)
            
            # Get weights used
            weights = self._get_effective_weights(available)
            components["weights"] = weights
            
            result["components"] = components
        
        if return_confidence:
            # Calculate confidence based on prediction variance across models
            pred_values = list(predictions.values())
            if len(pred_values) > 1:
                pred_std = np.std(pred_values, axis=0)
            else:
                pred_std = np.std(pred_values[0]) if len(pred_values) > 0 else 0.0
            
            confidence_interval = self._estimate_confidence_interval(
                ensemble_pred, pred_std
            )
            lower = confidence_interval[0]
            upper = confidence_interval[1]
            
            result["confidence"] = {
                "lower": to_python_value(lower) if isinstance(lower, np.ndarray) else float(lower),
                "upper": to_python_value(upper) if isinstance(upper, np.ndarray) else float(upper),
                "std": float(pred_std) if np.isscalar(pred_std) else (float(pred_std.item()) if hasattr(pred_std, 'item') else float(pred_std))
            }
        
        return result
    
    def _normalize_prediction(self, pred: np.ndarray) -> np.ndarray:
        """Normalize prediction to 1D array."""
        if pred.ndim == 0:
            return np.array([pred])
        elif pred.ndim > 1:
            return pred.flatten()
        return pred
    
    def _ensemble_predictions(
        self,
        predictions: Dict[str, np.ndarray],
        available: Dict[str, bool]
    ) -> np.ndarray:
        """
        Combine predictions from multiple models with weights.
        
        Args:
            predictions: Dictionary of model_name -> predictions
            available: Dictionary of model_name -> availability
            
        Returns:
            Ensemble prediction
        """
        # Get effective weights (normalized for available models)
        weights = self._get_effective_weights(available)
        
        # Weighted average
        ensemble = None
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 0.0)
            if ensemble is None:
                ensemble = weight * pred
            else:
                ensemble += weight * pred
        
        return ensemble
    
    def _get_effective_weights(self, available: Dict[str, bool]) -> Dict[str, float]:
        """Get normalized weights for available models."""
        weights = {
            "lightgbm": self.lgb_weight if available["lightgbm"] else 0.0,
            "xgboost": self.xgb_weight if available["xgboost"] else 0.0,
            "lstm": self.lstm_weight if available["lstm"] else 0.0,
            "neural_network": self.nn_weight if available["neural_network"] else 0.0
        }
        
        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            # Equal weights if all zero
            count = sum(available.values())
            if count > 0:
                weights = {k: (1.0 / count if available[k] else 0.0) for k in weights.keys()}
        
        return weights
    
    def _get_feature_names(self) -> Optional[List[str]]:
        """Get feature names from model metadata."""
        try:
            version = self.model_registry.current_version
            if version:
                model_info = self.model_registry.get_model_info(version)
                
                # Try XGBoost first (most likely to have feature names)
                if model_info and "xgboost" in model_info:
                    xgb_meta = model_info["xgboost"].get("metadata", {})
                    feature_names = xgb_meta.get("feature_names")
                    if feature_names:
                        return feature_names
                
                # Try LightGBM
                if model_info and "lightgbm" in model_info:
                    lgb_meta = model_info["lightgbm"].get("metadata", {})
                    feature_names = lgb_meta.get("feature_names")
                    if feature_names:
                        return feature_names
                
                # Try LSTM
                if model_info and "lstm" in model_info:
                    lstm_meta = model_info["lstm"].get("metadata", {})
                    # LSTM might not have feature names, but check anyway
                    feature_names = lstm_meta.get("feature_names")
                    if feature_names:
                        return feature_names
        except Exception as e:
            logger.warning(f"Could not get feature names from model metadata: {e}")
        
        # Fallback to feature engineer
        try:
            feature_names = self.feature_engineer.get_feature_names()
            if feature_names:
                return feature_names
        except Exception as e:
            logger.warning(f"Could not get feature names from feature engineer: {e}")
        
        return None
    
    def _get_expected_feature_count(self) -> Optional[int]:
        """Get expected feature count from model metadata."""
        try:
            version = self.model_registry.current_version
            if version:
                model_info = self.model_registry.get_model_info(version)
                
                # Try LightGBM first
                if model_info and "lightgbm" in model_info:
                    lgb_meta = model_info["lightgbm"].get("metadata", {})
                    feature_names = lgb_meta.get("feature_names", [])
                    if feature_names:
                        return len(feature_names)
                
                # Try XGBoost
                if model_info and "xgboost" in model_info:
                    xgb_meta = model_info["xgboost"].get("metadata", {})
                    feature_names = xgb_meta.get("feature_names", [])
                    if feature_names:
                        return len(feature_names)
                
                # Try LSTM
                if model_info and "lstm" in model_info:
                    lstm_meta = model_info["lstm"].get("metadata", {})
                    input_dim = lstm_meta.get("input_dim")
                    if input_dim:
                        return input_dim
                
                # Try Neural Network
                if model_info and "neural_network" in model_info:
                    nn_meta = model_info["neural_network"].get("metadata", {})
                    input_dim = nn_meta.get("input_dim")
                    if input_dim:
                        return input_dim
        except Exception as e:
            logger.debug(f"Could not get expected feature count from metadata: {e}")
        
        return None
    
    def _estimate_confidence_interval(
        self,
        predictions: np.ndarray,
        std: float,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate confidence intervals.
        
        Args:
            predictions: Point predictions
            std: Standard deviation estimate
            alpha: Significance level (0.05 = 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        z_score = 1.96  # 95% confidence
        margin = z_score * std
        
        lower = predictions - margin
        upper = predictions + margin
        
        return lower, upper
    
    def update_weights(
        self,
        lgb_weight: Optional[float] = None,
        xgb_weight: Optional[float] = None,
        lstm_weight: Optional[float] = None,
        nn_weight: Optional[float] = None
    ) -> None:
        """
        Update ensemble weights dynamically.
        
        Args:
            lgb_weight: New LightGBM weight
            xgb_weight: New XGBoost weight
            lstm_weight: New LSTM weight
            nn_weight: New Neural Network weight
        """
        if lgb_weight is not None:
            self.lgb_weight = lgb_weight
        if xgb_weight is not None:
            self.xgb_weight = xgb_weight
        if lstm_weight is not None:
            self.lstm_weight = lstm_weight
        if nn_weight is not None:
            self.nn_weight = nn_weight
        
        # Normalize
        total = self.lgb_weight + self.xgb_weight + self.lstm_weight + self.nn_weight
        if total > 0:
            self.lgb_weight /= total
            self.xgb_weight /= total
            self.lstm_weight /= total
            self.nn_weight /= total
        else:
            logger.warning("Weights sum to zero, keeping current weights")
