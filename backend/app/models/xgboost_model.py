"""
XGBoost model wrapper for stock prediction.
Handles model loading, prediction, and versioning.
"""
import xgboost as xgb
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class XGBoostModel:
    """Wrapper for XGBoost model."""
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize XGBoost model.
        
        Args:
            model_path: Path to saved XGBoost model (.pkl or native format)
        """
        self.model_path = model_path
        self.model: Optional[xgb.Booster] = None
        self.version: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
        self.feature_names: Optional[List[str]] = None
    
    def load(self, model_path: Optional[Path] = None) -> None:
        """
        Load XGBoost model from file.
        
        Args:
            model_path: Path to model file. If None, uses self.model_path
        """
        path = model_path or self.model_path
        if path is None:
            raise ValueError("No model path provided")
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        logger.info(f"Loading XGBoost model from {path}")
        
        # Try loading as native XGBoost format first
        try:
            self.model = xgb.Booster()
            self.model.load_model(str(path))
            logger.info("Loaded XGBoost model (native format)")
        except:
            # Fallback to pickle
            try:
                with open(path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Loaded XGBoost model (pickle format)")
            except Exception as e:
                raise ValueError(f"Failed to load model: {e}")
        
        # Try to load metadata if available
        # Try multiple possible metadata file names
        metadata_paths = [
            path.parent / f"{path.stem}_metadata.pkl",  # xgboost_model_metadata.pkl
            path.parent / "xgboost_metadata.pkl",  # xgboost_metadata.pkl (alternative naming)
            path.parent / f"{path.name}_metadata.pkl"  # xgboost_model.json_metadata.pkl (unlikely)
        ]
        
        metadata_loaded = False
        for metadata_path in metadata_paths:
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'rb') as f:
                        self.metadata = pickle.load(f)
                    self.version = self.metadata.get('version', 'unknown')
                    metadata_loaded = True
                    logger.info(f"Loaded XGBoost metadata from {metadata_path}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
        
        if not metadata_loaded:
            logger.warning(f"No metadata file found for XGBoost model at {path}. Feature names may not be available.")
        
        # Store feature names from metadata for prediction
        self.feature_names = self.metadata.get('feature_names', None)
        if self.feature_names:
            logger.info(f"XGBoost model loaded with {len(self.feature_names)} feature names")
        else:
            logger.warning("XGBoost model loaded without feature names. Predictions may fail if model requires feature names.")
    
    def predict(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Optional list of feature names. If None, uses stored feature names from metadata.
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        # Use provided feature names or fall back to stored ones
        names = feature_names or self.feature_names
        
        # Convert to DMatrix for XGBoost
        # If feature names are available, pass them to DMatrix
        if names:
            if len(names) != X.shape[1]:
                raise ValueError(
                    f"Feature count mismatch: model expects {len(names)} features "
                    f"(names: {names}), but received {X.shape[1]} features"
                )
            # Create DMatrix with feature names
            dmatrix = xgb.DMatrix(X, feature_names=names)
        else:
            # Fallback: create DMatrix without feature names
            # This may fail if model was trained with feature names
            logger.warning("No feature names available. XGBoost may fail if model was trained with feature names.")
            dmatrix = xgb.DMatrix(X)
        
        predictions = self.model.predict(dmatrix)
        return predictions
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: 'gain', 'weight', or 'cover'
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Get feature names from metadata or use indices
        feature_names = self.metadata.get('feature_names', 
                                         [f'feature_{i}' for i in range(self.model.num_feature())])
        
        scores = self.model.get_score(importance_type=importance_type)
        
        # Map to feature names
        importance = {}
        for key, value in scores.items():
            # XGBoost uses f0, f1, f2... format
            if key.startswith('f'):
                idx = int(key[1:])
                if idx < len(feature_names):
                    importance[feature_names[idx]] = value
        
        return importance

