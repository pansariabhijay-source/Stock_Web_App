"""
Data loading utilities for frontend.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def load_stock_data(file_path: Path) -> pd.DataFrame:
    """Load stock data from CSV."""
    df = pd.read_csv(file_path)
    
    # Parse date if present
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    
    return df


def prepare_features(df: pd.DataFrame, exclude_cols: list = None) -> tuple:
    """
    Prepare features for prediction.
    
    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    if exclude_cols is None:
        exclude_cols = ["Target_Close", "Date", "date"]
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    features = df[feature_cols].values
    
    return features, feature_cols

