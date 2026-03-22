"""
LightGBM model wrapper for stock prediction.
Production-grade implementation with proper error handling.
"""
import lightgbm as lgb
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LightGBMModel:
    """Wrapper for LightGBM model with production features."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize LightGBM model.
        
        Args:
            model_path: Path to saved LightGBM model (.txt or .pkl)
        """
        self.model_path = model_path
        self.model: Optional[lgb.Booster] = None
        self.version: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
    
    def load(self, model_path: Optional[Path] = None) -> None:
        """
        Load LightGBM model from file.
        
        Args:
            model_path: Path to model file. If None, uses self.model_path
        """
        path = model_path or self.model_path
        if path is None:
            raise ValueError("No model path provided")
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        logger.info(f"Loading LightGBM model from {path}")
        
        # Try loading as native LightGBM format first
        try:
            self.model = lgb.Booster(model_file=str(path))
            logger.info("Loaded LightGBM model (native format)")
        except Exception as e:
            # Fallback to pickle
            try:
                with open(path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Loaded LightGBM model (pickle format)")
            except Exception as e2:
                raise ValueError(f"Failed to load model: {e}, {e2}")
        
        # Try to load metadata if available
        metadata_path = path.parent / f"{path.stem}_metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
                self.version = self.metadata.get('version', 'unknown')
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        predictions = self.model.predict(X)
        return np.array(predictions)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: 'gain', 'split', or 'gain'
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Get feature names from metadata or use indices
        feature_names = self.metadata.get('feature_names', 
                                         [f'feature_{i}' for i in range(self.model.num_feature())])
        
        importance = self.model.feature_importance(importance_type=importance_type)
        
        # Map to feature names
        importance_dict = {}
        for i, score in enumerate(importance):
            if i < len(feature_names):
                importance_dict[feature_names[i]] = float(score)
        
        return importance_dict

