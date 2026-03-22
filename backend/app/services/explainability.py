"""
SHAP-based explainability service.
Provides feature importance and prediction explanations.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

logger = logging.getLogger(__name__)


class ExplainabilityService:
    """
    Service for model explainability using SHAP values.
    
    Provides:
    - Feature importance rankings
    - Prediction-level explanations
    - Feature contribution analysis
    """
    
    def __init__(self, model_registry):
        """
        Initialize explainability service.
        
        Args:
            model_registry: Model registry instance
        """
        self.model_registry = model_registry
        self.explainer: Optional[Any] = None
        self.feature_names: List[str] = []
    
    def initialize_explainer(self, X_sample: np.ndarray, feature_names: List[str]) -> None:
        """
        Initialize SHAP explainer with sample data.
        
        Args:
            X_sample: Sample feature matrix for background
            feature_names: List of feature names
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library not installed")
        
        if self.model_registry.xgb_model is None:
            raise ValueError("XGBoost model not loaded")
        
        self.feature_names = feature_names
        
        # Create SHAP explainer for XGBoost
        # Use TreeExplainer for XGBoost (fast and exact)
        model = self.model_registry.xgb_model.model
        self.explainer = shap.TreeExplainer(model)
        
        logger.info("SHAP explainer initialized")
    
    def explain_prediction(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP values.
        
        Args:
            features: Feature vector (1, n_features) or (n_features,)
            feature_names: Feature names. Uses stored names if None.
            
        Returns:
            Dictionary with SHAP values and explanations
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")
        
        if not SHAP_AVAILABLE:
            # Fallback to feature importance
            return self._fallback_explanation(features, feature_names)
        
        # Ensure 2D array
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(features)
        
        # Get feature names
        names = feature_names or self.feature_names
        if not names:
            names = [f"feature_{i}" for i in range(features.shape[1])]
        
        # Create explanation dictionary
        explanation = {
            "shap_values": shap_values[0].tolist() if len(shap_values) > 0 else shap_values.tolist(),
            "feature_names": names,
            "base_value": float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else None,
            "feature_importance": self._get_feature_importance_from_shap(shap_values, names)
        }
        
        return explanation
    
    def _get_feature_importance_from_shap(
        self,
        shap_values: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Calculate feature importance from SHAP values (mean absolute SHAP)."""
        if shap_values.ndim > 1:
            importance = np.abs(shap_values).mean(axis=0)
        else:
            importance = np.abs(shap_values)
        
        return dict(zip(feature_names, importance.tolist()))
    
    def _fallback_explanation(
        self,
        features: np.ndarray,
        feature_names: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Fallback explanation using model feature importance."""
        if self.model_registry.xgb_model is None:
            raise ValueError("Model not available for fallback explanation")
        
        importance = self.model_registry.xgb_model.get_feature_importance()
        
        names = feature_names or list(importance.keys())
        
        return {
            "feature_importance": importance,
            "feature_names": names,
            "note": "Using feature importance (SHAP not available)"
        }
    
    def get_global_importance(self, X_sample: np.ndarray) -> Dict[str, float]:
        """
        Get global feature importance across a sample.
        
        Args:
            X_sample: Sample feature matrix
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.explainer is None:
            # Use model's built-in importance
            if self.model_registry.xgb_model:
                return self.model_registry.xgb_model.get_feature_importance()
            return {}
        
        if not SHAP_AVAILABLE:
            return self._fallback_explanation(X_sample[0:1], None)["feature_importance"]
        
        # Calculate SHAP values for sample
        shap_values = self.explainer.shap_values(X_sample)
        
        # Average absolute SHAP values
        if shap_values.ndim > 1:
            importance = np.abs(shap_values).mean(axis=0)
        else:
            importance = np.abs(shap_values)
        
        names = self.feature_names or [f"feature_{i}" for i in range(len(importance))]
        
        return dict(zip(names, importance.tolist()))

