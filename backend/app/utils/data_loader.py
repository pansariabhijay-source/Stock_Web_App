"""
Data loading utilities for stock price data.
Handles CSV loading, preprocessing, and validation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and preprocesses stock price data."""
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to CSV file. If None, uses default from config.
        """
        self.data_path = data_path
        self._data: Optional[pd.DataFrame] = None
    
    def load_raw_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load raw stock data from CSV.
        
        Expected columns:
        - Date, Open Price, High Price, Low Price, Close Price, 
          Total Traded Quantity, Turnover, etc.
        
        Returns:
            DataFrame with Date as index
        """
        path = file_path or self.data_path
        if path is None:
            raise ValueError("No data path provided")
        
        if not Path(path).exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path, index_col=0)  # Read first column as index
        
        # Clean column names
        df.columns = [c.strip() for c in df.columns]
        
        # Try to parse index as date only if it looks like dates
        if not isinstance(df.index, pd.DatetimeIndex):
            # Check if index contains date-like strings (contains '-' or '/' or is numeric in date format)
            sample_val = str(df.index[0]) if len(df.index) > 0 else ""
            is_date_like = ('-' in sample_val and len(sample_val) > 5) or ('/' in sample_val) or (sample_val.isdigit() and len(sample_val) == 8)  # YYYYMMDD format
            
            if is_date_like:
                try:
                    df.index = pd.to_datetime(df.index, errors='coerce')
                    logger.info("Parsed index as dates")
                except (ValueError, TypeError):
                    logger.warning("Could not parse index as dates, keeping as-is")
                    pass
            else:
                # Integer index or other non-date index - keep as is
                logger.info(f"Using {type(df.index).__name__} index (not dates)")
        
        df = df.sort_index()  # Sort by index
        
        # Ensure numeric types
        numeric_cols = [
            "Open Price", "High Price", "Low Price", "Last Price", 
            "Close Price", "Average Price", "Total Traded Quantity",
            "Turnover", "No. of Trades", "Deliverable Qty"
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing values (time-series safe)
        price_cols = [c for c in numeric_cols if c in df.columns and "Price" in c]
        if price_cols:
            df[price_cols] = df[price_cols].ffill().bfill()
        
        # Fill remaining with median
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        self._data = df
        return df
    
    def get_latest_data(self, n_days: int = 30) -> pd.DataFrame:
        """
        Get the most recent N days of data.
        
        Args:
            n_days: Number of days to retrieve
            
        Returns:
            DataFrame with latest N days
        """
        if self._data is None:
            raise ValueError("Data not loaded. Call load_raw_data() first.")
        
        return self._data.tail(n_days).copy()
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "issues": []
        }
        
        required_cols = ["Close Price", "Open Price", "High Price", "Low Price"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            results["valid"] = False
            results["issues"].append(f"Missing required columns: {missing}")
        
        if df.empty:
            results["valid"] = False
            results["issues"].append("DataFrame is empty")
        
        # Check for excessive missing values
        if "Close Price" in df.columns:
            missing_pct = df["Close Price"].isna().sum() / len(df)
            if missing_pct > 0.1:
                results["valid"] = False
                results["issues"].append(f"High missing values in Close Price: {missing_pct:.2%}")
        
        return results

