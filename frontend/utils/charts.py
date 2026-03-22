"""
Chart utilities for visualization using Plotly.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional, Dict


def plot_candlestick_with_predictions(
    df: pd.DataFrame,
    predictions: Optional[List[float]] = None,
    confidence_lower: Optional[List[float]] = None,
    confidence_upper: Optional[List[float]] = None
) -> go.Figure:
    """
    Plot candlestick chart with predictions overlay.
    
    Args:
        df: DataFrame with OHLC data
        predictions: Prediction values
        confidence_lower: Lower confidence bound
        confidence_upper: Upper confidence bound
    """
    fig = go.Figure()
    
    # Candlestick
    if all(col in df.columns for col in ["Open Price", "High Price", "Low Price", "Close Price"]):
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open Price"],
            high=df["High Price"],
            low=df["Low Price"],
            close=df["Close Price"],
            name="Price"
        ))
    
    # Predictions
    if predictions:
        fig.add_trace(go.Scatter(
            x=df.index[-len(predictions):] if len(predictions) < len(df) else df.index,
            y=predictions,
            mode="lines+markers",
            name="Prediction",
            line=dict(color="green", width=2)
        ))
    
    # Confidence bands
    if confidence_lower and confidence_upper:
        x_vals = df.index[-len(confidence_lower):] if len(confidence_lower) < len(df) else df.index
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=confidence_upper,
            mode="lines",
            name="Upper Bound",
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=confidence_lower,
            mode="lines",
            name="Confidence Interval",
            fill="tonexty",
            fillcolor="rgba(0,100,80,0.2)",
            line=dict(width=0)
        ))
    
    fig.update_layout(
        title="Stock Price with Predictions",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        height=500
    )
    
    return fig


def plot_equity_curve(equity_curve: List[float], dates: Optional[List] = None) -> go.Figure:
    """Plot equity curve from backtesting."""
    fig = go.Figure()
    
    x_vals = dates if dates else list(range(len(equity_curve)))
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=equity_curve,
        mode="lines",
        name="Equity Curve",
        line=dict(color="blue", width=2)
    ))
    
    # Add horizontal line at 1.0
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Initial Capital")
    
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date" if dates else "Time",
        yaxis_title="Equity",
        hovermode="x unified",
        height=400
    )
    
    return fig


def plot_drawdown(drawdown: List[float], dates: Optional[List] = None) -> go.Figure:
    """Plot drawdown series."""
    fig = go.Figure()
    
    x_vals = dates if dates else list(range(len(drawdown)))
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=[d * 100 for d in drawdown],  # Convert to percentage
        mode="lines",
        fill="tozeroy",
        name="Drawdown",
        line=dict(color="red", width=2),
        fillcolor="rgba(255,0,0,0.3)"
    ))
    
    fig.update_layout(
        title="Drawdown Analysis",
        xaxis_title="Date" if dates else "Time",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        height=400
    )
    
    return fig


def plot_predictions_comparison(
    actuals: List[float],
    predictions: List[float],
    dates: Optional[List] = None
) -> go.Figure:
    """Plot predictions vs actuals."""
    fig = go.Figure()
    
    x_vals = dates if dates else list(range(len(actuals)))
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=actuals,
        mode="lines+markers",
        name="Actual",
        line=dict(color="blue", width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=predictions,
        mode="lines+markers",
        name="Predicted",
        line=dict(color="red", width=2, dash="dash")
    ))
    
    fig.update_layout(
        title="Predictions vs Actual",
        xaxis_title="Date" if dates else "Time",
        yaxis_title="Price",
        hovermode="x unified",
        height=400
    )
    
    return fig


def plot_shap_waterfall(
    shap_values: List[float],
    feature_names: List[str],
    base_value: float = 0
) -> go.Figure:
    """Plot SHAP waterfall chart."""
    # Sort by absolute SHAP value
    data = list(zip(feature_names, shap_values))
    data.sort(key=lambda x: abs(x[1]), reverse=True)
    
    feature_names_sorted = [d[0] for d in data]
    shap_values_sorted = [d[1] for d in data]
    
    # Calculate cumulative values
    cumulative = base_value
    y_start = [cumulative]
    y_end = []
    
    for val in shap_values_sorted:
        cumulative += val
        y_end.append(cumulative)
        y_start.append(cumulative)
    
    y_start = y_start[:-1]
    
    # Create waterfall
    fig = go.Figure()
    
    colors = ["green" if v > 0 else "red" for v in shap_values_sorted]
    
    fig.add_trace(go.Waterfall(
        orientation="v",
        measure=["absolute"] + ["relative"] * (len(shap_values_sorted) - 1) + ["total"],
        x=feature_names_sorted + ["Prediction"],
        y=[base_value] + shap_values_sorted + [y_end[-1]],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title="SHAP Waterfall Plot",
        xaxis_title="Feature",
        yaxis_title="SHAP Value",
        height=500
    )
    
    return fig


def plot_feature_importance(importance: Dict[str, float]) -> go.Figure:
    """Plot feature importance bar chart."""
    # Sort by importance
    sorted_items = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)
    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    colors = ["green" if v > 0 else "red" for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color=colors,
        name="Importance"
    ))
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=600,
        yaxis={"autorange": "reversed"}
    )
    
    return fig

