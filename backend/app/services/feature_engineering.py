"""
Feature engineering pipeline for stock prediction.
Creates technical indicators, lag features, and rolling statistics.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering service for stock price prediction.
    
    Creates:
    - Technical indicators (MA, RSI, MACD, Bollinger Bands)
    - Lag features
    - Rolling statistics
    - Volume-based features
    - Price-based features (returns, volatility)
    """
    
    def __init__(self, feature_version: str = "v1"):
        """
        Initialize feature engineer.
        
        Args:
            feature_version: Version identifier for feature set
        """
        self.feature_version = feature_version
        self.feature_names: List[str] = []
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features from raw OHLCV data.
        
        Args:
            df: DataFrame with Date index and OHLCV columns
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering")
        df = df.copy()
        
        # Ensure Close Price exists
        if "Close Price" not in df.columns:
            raise ValueError("Close Price column required")
        
        close = df["Close Price"]
        
        # 1. Price-based features
        df = self._add_price_features(df, close)
        
        # 2. Technical indicators
        df = self._add_technical_indicators(df, close)
        
        # 3. Lag features
        df = self._add_lag_features(df, close)
        
        # 4. Volume features
        df = self._add_volume_features(df)
        
        # 5. Rolling statistics
        df = self._add_rolling_stats(df, close)
        
        # Fill NaN values introduced by shifts/rolling
        df = df.ffill().bfill()
        
        # Store feature names (exclude target and date columns)
        self.feature_names = [
            c for c in df.columns 
            if c not in ["Target_Close", "Date"] and df[c].dtype in [np.float64, np.int64]
        ]
        
        logger.info(f"Created {len(self.feature_names)} features")
        return df
    
    def _add_price_features(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Add price-based features."""
        # Returns
        df["Return"] = close.pct_change()
        df["Return_abs"] = df["Return"].abs()
        
        # Price ratios
        if "High Price" in df.columns and "Low Price" in df.columns:
            df["HL_Ratio"] = df["High Price"] / df["Low Price"]
            df["OC_Ratio"] = df["Open Price"] / close if "Open Price" in df.columns else None
        
        # Volatility (rolling std of returns)
        for window in [7, 14, 21]:
            df[f"Volatility_{window}"] = df["Return"].rolling(window=window, min_periods=1).std()
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Add technical indicators."""
        # Moving Averages
        for window in [7, 14, 21, 50]:
            df[f"MA_{window}"] = close.rolling(window=window, min_periods=1).mean()
            df[f"MA_{window}_ratio"] = close / df[f"MA_{window}"]
        
        # RSI (Relative Strength Index)
        df["RSI_14"] = self._calculate_rsi(close, window=14)
        
        # MACD
        macd, signal = self._calculate_macd(close)
        df["MACD"] = macd
        df["MACD_signal"] = signal
        df["MACD_histogram"] = macd - signal
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(close, window=20)
        df["BB_upper"] = bb_upper
        df["BB_lower"] = bb_lower
        df["BB_width"] = (bb_upper - bb_lower) / bb_middle
        df["BB_position"] = (close - bb_lower) / (bb_upper - bb_lower)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Add lagged price features."""
        for lag in [1, 2, 3, 5, 7]:
            df[f"Close_lag_{lag}"] = close.shift(lag)
            df[f"Close_lag_{lag}_ratio"] = close / df[f"Close_lag_{lag}"]
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        if "Total Traded Quantity" not in df.columns:
            return df
        
        volume = df["Total Traded Quantity"]
        
        # Volume change
        df["Volume_change"] = volume.pct_change().fillna(0)
        df["Volume_abs_change"] = volume.diff().abs()
        
        # Volume moving averages
        for window in [7, 14, 21]:
            df[f"Volume_MA_{window}"] = volume.rolling(window=window, min_periods=1).mean()
            df[f"Volume_ratio_{window}"] = volume / df[f"Volume_MA_{window}"]
        
        # Price-Volume relationship
        if "Return" in df.columns:
            df["Price_Volume"] = df["Return"] * df["Volume_change"]
        
        return df
    
    def _add_rolling_stats(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Add rolling statistics."""
        for window in [7, 14, 21]:
            df[f"STD_{window}"] = close.rolling(window=window, min_periods=1).std()
            df[f"MIN_{window}"] = close.rolling(window=window, min_periods=1).min()
            df[f"MAX_{window}"] = close.rolling(window=window, min_periods=1).max()
            df[f"Range_{window}"] = df[f"MAX_{window}"] - df[f"MIN_{window}"]
            
            # Momentum
            df[f"Momentum_{window}"] = close - close.shift(window)
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Neutral RSI when insufficient data
    
    @staticmethod
    def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD and signal line."""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd.fillna(0), macd_signal.fillna(0)
    
    @staticmethod
    def _calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> tuple:
        """Calculate Bollinger Bands."""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper = rolling_mean + (rolling_std * num_std)
        lower = rolling_mean - (rolling_std * num_std)
        return upper.fillna(prices), lower.fillna(prices), rolling_mean.fillna(prices)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names.copy()
    
    def select_features(self, df: pd.DataFrame, feature_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Select specific features from dataframe.
        
        Args:
            df: DataFrame with all features
            feature_list: List of feature names. If None, uses recommended set.
            
        Returns:
            DataFrame with selected features
        """
        if feature_list is None:
            # Recommended feature set based on MI analysis
            feature_list = [
                "Open Price", "High Price", "Low Price", "Close Price",
                "Close_lag_1", "Close_lag_2",
                "MA_7", "MA_21",
                "Total Traded Quantity", "Volume_change", "Turnover",
                "RSI_14", "MACD", "BB_position",
                "Volatility_14", "Momentum_14"
            ]
        
        # Only select features that exist
        available_features = [f for f in feature_list if f in df.columns]
        return df[available_features].copy()

