"""
LSTM/GRU model for time-series stock prediction.
Deep learning architecture optimized for sequential data.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import pickle
import logging

logger = logging.getLogger(__name__)


class LSTMModel(nn.Module):
    """
    LSTM model for stock price prediction.
    Uses bidirectional LSTM with attention mechanism.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True,
        bidirectional: bool = True
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_dim, lstm_output_dim),
                nn.Tanh(),
                nn.Linear(lstm_output_dim, 1)
            )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim) or (batch_size, input_dim)
        """
        # Handle 2D input (single timestep) by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply attention if enabled
        if self.use_attention:
            attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
            attention_weights = torch.softmax(attention_weights, dim=1)
            attended = (lstm_out * attention_weights).sum(dim=1)  # (batch_size, hidden_dim)
        else:
            # Use last timestep
            attended = lstm_out[:, -1, :]
        
        # Final prediction
        output = self.fc(attended)
        return output.squeeze(-1)


class LSTMStockModel:
    """Wrapper for PyTorch LSTM model."""
    
    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None):
        """
        Initialize LSTM model wrapper.
        
        Args:
            model_path: Path to saved model
            device: 'cpu' or 'cuda'. Auto-detects if None.
        """
        self.model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[LSTMModel] = None
        self.scaler: Optional[Any] = None  # StandardScaler
        self.version: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
    
    def load(self, model_path: Optional[Path] = None) -> None:
        """
        Load PyTorch LSTM model and scaler.
        
        Args:
            model_path: Path to model directory. If None, uses self.model_path
        """
        path = Path(model_path or self.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model path not found: {path}")
        
        logger.info(f"Loading LSTM model from {path}")
        
        # Load metadata to get model architecture
        metadata_file = path / "metadata.pkl" if path.is_dir() else path.parent / f"{path.stem}_metadata.pkl"
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
                self.version = self.metadata.get('version', 'unknown')
                input_dim = self.metadata.get('input_dim')
                hidden_dim = self.metadata.get('hidden_dim', 128)
                num_layers = self.metadata.get('num_layers', 2)
                use_attention = self.metadata.get('use_attention', True)
                bidirectional = self.metadata.get('bidirectional', True)
        else:
            # Default architecture
            input_dim = 16  # Based on feature set
            hidden_dim = 128
            num_layers = 2
            use_attention = True
            bidirectional = True
            logger.warning("Metadata not found, using default architecture")
        
        # Load model state
        model_file = path / "model.pth" if path.is_dir() else path
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Initialize and load model
        self.model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_attention=use_attention,
            bidirectional=bidirectional
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.model.eval()
        
        # Load scaler
        if path.is_dir():
            scaler_file = path / "scaler.pkl"
        else:
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
            Predictions array
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

