"""
API client for communicating with FastAPI backend.
"""
import requests
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class APIClient:
    """Client for Stock Prediction API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    def predict(
        self,
        features: List[float],
        symbol: Optional[str] = None,
        return_components: bool = False,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction.
        
        Args:
            features: Feature vector
            symbol: Stock symbol (optional)
            return_components: Return individual model predictions
            return_confidence: Return confidence intervals
            
        Returns:
            Prediction result
        """
        payload = {
            "features": features,
            "return_components": return_components,
            "return_confidence": return_confidence
        }
        
        if symbol:
            payload["symbol"] = symbol
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/predict",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def backtest(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        train_size: float = 0.8
    ) -> Dict[str, Any]:
        """
        Run backtest.
        
        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            train_size: Training set proportion
            
        Returns:
            Backtest results
        """
        payload = {
            "train_size": train_size
        }
        
        if start_date:
            payload["start_date"] = start_date
        if end_date:
            payload["end_date"] = end_date
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/backtest",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    def explain(self, features: List[float]) -> Dict[str, Any]:
        """
        Get SHAP explanation.
        
        Args:
            features: Feature vector
            
        Returns:
            Explanation result
        """
        payload = {"features": features}
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/explain",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            raise
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models.
        
        Returns:
            List of model information
        """
        try:
            response = self.session.get(f"{self.base_url}/api/v1/models")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise
    
    def get_active_model_info(self) -> Dict[str, Any]:
        """
        Get information about the active model including feature requirements.
        
        Returns:
            Active model information including feature names and count
        """
        try:
            response = self.session.get(f"{self.base_url}/api/v1/model/info")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get active model info: {e}")
            raise
    
    def train_models(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train models with new data.
        
        Args:
            data: List of dictionaries containing stock data
            
        Returns:
            Training results including new model version
        """
        payload = {"data": data}
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/train",
                json=payload,
                timeout=600  # 10 minute timeout for training
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

