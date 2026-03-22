"""
Neural Network model wrapper for stock prediction.
PyTorch-based residual learner.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import pickle
import logging

logger = logging.getLogger(__name__)


class ResidualNN(nn.Module):
    """Neural network for learning XGBoost residuals."""
    
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32], dropout: float = 0.1):
        """
        Initialize residual neural network.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        return self.net(x).squeeze()


class NeuralNetworkModel:
    """Wrapper for PyTorch neural network model."""
    
    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None):
        """
        Initialize neural network model.
        
        Args:
            model_path: Path to saved model
            device: 'cpu' or 'cuda'. Auto-detects if None.
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[ResidualNN] = None
        self.scaler: Optional[Any] = None  # StandardScaler
        self.version: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
    
    def load(self, model_path: Optional[Path] = None) -> None:
        """
        Load PyTorch model and scaler.
        
        Args:
            model_path: Path to model directory. If None, uses self.model_path
        """
        path = Path(model_path or self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")
        
        logger.info(f"Loading neural network from {path}")
        
        # Load model state
        model_file = path / "model.pth" if path.is_dir() else path
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Load metadata to get model architecture
        metadata_file = path / "metadata.pkl" if path.is_dir() else path.parent / f"{path.stem}_metadata.pkl"
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
                self.version = self.metadata.get('version', 'unknown')
                input_dim = self.metadata.get('input_dim')
                hidden_dims = self.metadata.get('hidden_dims', [64, 32])
        else:
            # Default architecture
            input_dim = 11  # Based on feature set
            hidden_dims = [64, 32]
            logger.warning("Metadata not found, using default architecture")
        
        # Initialize and load model
        self.model = ResidualNN(input_dim, hidden_dims).to(self.device)
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.model.eval()
        
        # Load scaler
        if path.is_dir():
            scaler_file = path / "scaler.pkl"
        else:
            # If path is a file, scaler should be in the same directory
            scaler_file = path.parent / "scaler.pkl"
        
        if scaler_file.exists():
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"Loaded scaler from {scaler_file}")
        else:
            logger.warning(f"Scaler not found at {scaler_file}. Predictions may be inaccurate.")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predictions array (residuals)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        # Scale features if scaler available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Convert to tensor and predict
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        
        return predictions

