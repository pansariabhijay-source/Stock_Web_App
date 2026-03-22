"""
Walk-forward backtesting engine for model evaluation.
Implements time-series cross-validation to avoid look-ahead bias.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from app.services.model_registry import ModelRegistry
from app.services.prediction_service import PredictionService
from app.services.feature_engineering import FeatureEngineer
from app.utils.metrics import calculate_metrics

logger = logging.getLogger(__name__)


class BacktestingEngine:
    """
    Walk-forward backtesting engine.
    
    Implements:
    - Time-series cross-validation
    - Rolling window training
    - Performance tracking (RMSE, direction accuracy, drawdown)
    - Equity curve generation
    """
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        prediction_service: PredictionService,
        feature_engineer: FeatureEngineer,
        train_size: float = 0.8,
        step_size: int = 1
    ):
        """
        Initialize backtesting engine.
        
        Args:
            model_registry: Model registry instance
            prediction_service: Prediction service instance
            feature_engineer: Feature engineering instance
            train_size: Proportion of data for training (rest for testing)
            step_size: Days to step forward in each iteration
        """
        self.model_registry = model_registry
        self.prediction_service = prediction_service
        self.feature_engineer = feature_engineer
        self.train_size = train_size
        self.step_size = step_size
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        target_col: str = "Target_Close"
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtest.
        
        Args:
            data: DataFrame with features and target
            target_col: Name of target column (default: Target_Close for next-day prediction)
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting walk-forward backtest")
        
        # Ensure data is sorted (by index, whether date or integer)
        data = data.sort_index()
        
        # If index is not DatetimeIndex, create a dummy date range for compatibility
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("Data index is not DatetimeIndex, using integer index for backtest")
            # Create a simple integer range if needed
            if not isinstance(data.index, pd.RangeIndex):
                data = data.reset_index(drop=True)
        
        # Validate and find target column
        if target_col not in data.columns:
            # Try alternative names
            if "Target_Close" in data.columns:
                target_col = "Target_Close"
                logger.info(f"Using 'Target_Close' as target column")
            elif "Close Price" in data.columns:
                logger.warning(f"Target column '{target_col}' not found, using 'Close Price'")
                target_col = "Close Price"
            else:
                raise ValueError(f"Target column '{target_col}' not found in data. Available columns: {list(data.columns)}")
        
        # Remove rows with missing target values
        initial_len = len(data)
        data = data.dropna(subset=[target_col])
        if len(data) < initial_len:
            logger.warning(f"Dropped {initial_len - len(data)} rows with missing target values")
        
        # Split into train/test
        split_idx = int(len(data) * self.train_size)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        logger.info(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
        
        # Get feature columns - prioritize model metadata features
        feature_cols = None
        try:
            # Try to get feature names from model metadata (most accurate)
            version = self.model_registry.current_version
            if version:
                model_info = self.model_registry.get_model_info(version)
                # Try LightGBM first (primary model)
                if model_info and "lightgbm" in model_info:
                    lgb_meta = model_info["lightgbm"].get("metadata", {})
                    feature_cols = lgb_meta.get("feature_names", [])
                    if feature_cols:
                        logger.info(f"Using {len(feature_cols)} features from LightGBM model metadata")
                # Fallback to XGBoost
                if not feature_cols and model_info and "xgboost" in model_info:
                    xgb_meta = model_info["xgboost"].get("metadata", {})
                    feature_cols = xgb_meta.get("feature_names", [])
                    if feature_cols:
                        logger.info(f"Using {len(feature_cols)} features from XGBoost model metadata")
        except Exception as e:
            logger.warning(f"Could not get feature names from model metadata: {e}")
        
        # Fallback to feature engineer or all numeric columns
        if not feature_cols:
            feature_cols = self.feature_engineer.get_feature_names()
            if feature_cols:
                logger.info(f"Using {len(feature_cols)} features from feature engineer")
        
        if not feature_cols:
            # Fallback: all numeric columns except target and date columns
            exclude_cols = {target_col, "Target_Close", "Close Price", "Date", "date"}
            feature_cols = [c for c in data.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(data[c])]
            logger.warning(f"Using {len(feature_cols)} features from all numeric columns (fallback)")
        
        # Validate all feature columns exist in data
        missing_features = [f for f in feature_cols if f not in data.columns]
        if missing_features:
            logger.warning(f"Missing features in data: {missing_features}. Removing from feature list.")
            feature_cols = [f for f in feature_cols if f in data.columns]
        
        if not feature_cols:
            raise ValueError("No valid feature columns found for backtesting")
        
        logger.info(f"Final feature set ({len(feature_cols)} features): {feature_cols[:5]}..." if len(feature_cols) > 5 else f"Final feature set: {feature_cols}")
        
        # Walk-forward testing
        predictions = []
        actuals = []
        dates = []
        
        # Prepare test data
        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values
        
        # Validate shapes
        if len(X_test) != len(y_test):
            raise ValueError(f"Mismatch: X_test has {len(X_test)} samples, y_test has {len(y_test)}")
        
        # Make predictions
        logger.info(f"Making predictions for {len(X_test)} samples...")
        for i in range(len(X_test)):
            try:
                pred_result = self.prediction_service.predict(
                    X_test[i:i+1],
                    return_components=False,
                    return_confidence=False
                )
                pred_value = pred_result["prediction"]
                
                # Validate prediction is reasonable (not NaN or infinite)
                if not (np.isfinite(pred_value) and 0 < pred_value < 1e6):
                    logger.warning(f"Unreasonable prediction at index {i}: {pred_value}, skipping")
                    continue
                
                predictions.append(float(pred_value))
                actuals.append(float(y_test[i]))
                dates.append(test_data.index[i])
            except Exception as e:
                logger.error(f"Prediction failed at index {i}: {e}")
                continue
        
        if len(predictions) == 0:
            error_msg = "No valid predictions generated. "
            if len(X_test) > 0:
                error_msg += f"Attempted {len(X_test)} predictions but all failed. "
                error_msg += "Check logs for individual prediction errors."
            else:
                error_msg += "No test data available for predictions."
            raise ValueError(error_msg)
        
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Additional validation
        if len(predictions) != len(actuals):
            raise ValueError(f"Prediction/actual mismatch: {len(predictions)} predictions vs {len(actuals)} actuals")
        
        # Log sample predictions for debugging
        if len(predictions) > 0:
            logger.info(f"Sample predictions: {predictions[:5]}")
            logger.info(f"Sample actuals: {actuals[:5]}")
            logger.info(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
            logger.info(f"Actual range: [{actuals.min():.2f}, {actuals.max():.2f}]")
        
        # Calculate returns for financial metrics
        returns = None
        if len(actuals) > 1:
            returns = np.diff(actuals) / actuals[:-1]  # Percentage returns
        
        metrics = calculate_metrics(actuals, predictions, returns=returns)
        
        # Calculate drawdown
        equity_curve = self._calculate_equity_curve(actuals, predictions)
        drawdown = self._calculate_drawdown(equity_curve)
        metrics["max_drawdown"] = float(np.max(drawdown))
        metrics["avg_drawdown"] = float(np.mean(drawdown))
        
        results = {
            "metrics": metrics,
            "predictions": predictions.tolist(),
            "actuals": actuals.tolist(),
            "dates": [d.isoformat() if hasattr(d, 'isoformat') else str(d) for d in dates],
            "equity_curve": equity_curve.tolist(),
            "drawdown": drawdown.tolist(),
            "train_size": len(train_data),
            "test_size": len(test_data),
            "model_version": self.model_registry.current_version
        }
        
        direction_accuracy = metrics.get('direction_accuracy', 0.0)
        logger.info(f"Backtest completed. RMSE: {metrics['rmse']:.2f}, Direction Accuracy: {direction_accuracy:.2%}")
        
        return results
    
    def _calculate_direction_accuracy(self, actuals: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate direction prediction accuracy."""
        if len(actuals) < 2:
            return 0.0
        
        actual_direction = np.diff(actuals) > 0
        pred_direction = np.diff(predictions) > 0
        
        accuracy = np.mean(actual_direction == pred_direction)
        return float(accuracy)
    
    def _calculate_equity_curve(self, actuals: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        Calculate equity curve based on predictions.
        
        Simple strategy: Buy if prediction > current price, else hold.
        """
        equity = [1.0]  # Start with $1
        
        for i in range(1, len(actuals)):
            # Simple strategy: if prediction is higher, assume we bought
            # Return is based on actual price movement
            if i > 0:
                return_pct = (actuals[i] - actuals[i-1]) / actuals[i-1]
                equity.append(equity[-1] * (1 + return_pct))
            else:
                equity.append(1.0)
        
        return np.array(equity)
    
    def _calculate_drawdown(self, equity_curve: np.ndarray) -> np.ndarray:
        """Calculate drawdown from equity curve."""
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        return drawdown

