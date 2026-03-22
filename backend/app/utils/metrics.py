"""
Comprehensive performance metrics for model evaluation.
Includes regression metrics, financial metrics, and statistical measures.
"""
import numpy as np
from typing import Dict, Optional
import warnings
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, returns: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        returns: Optional returns series for financial metrics
        
    Returns:
        Dictionary with metric names and values
    """
    # Remove any NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {
            "rmse": float('nan'),
            "mae": float('nan'),
            "mape": float('nan'),
            "r2": float('nan')
        }
    
    # Basic regression metrics
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean(errors ** 2))
    
    # MAE (Mean Absolute Error)
    mae = np.mean(abs_errors)
    
    # MAPE (Mean Absolute Percentage Error) - more robust version
    # Avoid division by zero
    non_zero_mask = np.abs(y_true) > 1e-8
    if np.sum(non_zero_mask) > 0:
        mape = np.mean(np.abs(errors[non_zero_mask] / y_true[non_zero_mask])) * 100
    else:
        mape = float('nan')
    
    # R² (Coefficient of Determination)
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # Mean Error (Bias)
    mean_error = np.mean(errors)
    
    # Median Absolute Error (robust to outliers)
    medae = np.median(abs_errors)
    
    # Max Error
    max_error = np.max(abs_errors)
    
    # Error Standard Deviation
    error_std = np.std(errors)
    
    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "medae": float(medae),
        "mape": float(mape),
        "r2": float(r2),
        "mean_error": float(mean_error),
        "max_error": float(max_error),
        "error_std": float(error_std),
    }
    
    # Direction accuracy (for time series)
    if len(y_true) > 1:
        actual_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        direction_accuracy = np.mean(actual_direction == pred_direction)
        metrics["direction_accuracy"] = float(direction_accuracy)
    
    # Financial metrics if returns are provided
    if returns is not None and len(returns) > 0:
        # Returns array might be shorter (len-1) if calculated from np.diff
        # We need to align returns with the masked y_true/y_pred
        original_len = len(mask)
        
        if len(returns) == original_len:
            # Same length - apply mask directly
            returns = returns[mask]
        elif len(returns) == original_len - 1:
            # Returns is one shorter (from np.diff) - need to align with mask
            # Create mask for returns: remove last element and apply original mask
            if len(mask) > 1:
                returns_mask = mask[:-1]
                returns = returns[returns_mask]
            else:
                # Not enough data for returns after masking
                returns = None
        else:
            # Size mismatch - recalculate returns from masked actuals (most reliable)
            if len(y_true) > 1:
                returns = np.diff(y_true) / (y_true[:-1] + 1e-8)  # Add small epsilon to avoid division by zero
            else:
                returns = None
        
        if returns is not None and len(returns) > 1:
            # Sharpe Ratio (assuming daily returns, annualize with sqrt(252))
            sharpe_ratio = (np.mean(returns) / (np.std(returns) + 1e-8)) * np.sqrt(252)
            metrics["sharpe_ratio"] = float(sharpe_ratio)
            
            # Sortino Ratio (only penalizes downside volatility)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                sortino_ratio = (np.mean(returns) / (downside_std + 1e-8)) * np.sqrt(252)
                metrics["sortino_ratio"] = float(sortino_ratio)
    
    # Additional time-series specific metrics
    if len(y_true) > 1:
        # Mean Absolute Scaled Error (MASE) - scale by naive forecast error
        naive_forecast_errors = np.abs(np.diff(y_true))
        if np.mean(naive_forecast_errors) > 1e-8:
            mase = np.mean(abs_errors[1:]) / np.mean(naive_forecast_errors)
            metrics["mase"] = float(mase)
        
        # Symmetric MAPE (sMAPE) - better for forecasts near zero
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        non_zero_denom = denominator > 1e-8
        if np.sum(non_zero_denom) > 0:
            smape = np.mean(np.abs(errors[non_zero_denom] / denominator[non_zero_denom])) * 100
            metrics["smape"] = float(smape)
    
    return metrics

