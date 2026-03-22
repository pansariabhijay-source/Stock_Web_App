"""
Training service for retraining models with new data.
Provides API-accessible training functionality.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from app.services.feature_engineering import FeatureEngineer
from app.services.model_registry import ModelRegistry
from app.config import settings
from app.models.lstm_model import LSTMModel

logger = logging.getLogger(__name__)

# Configuration
SEED = 42
TEST_RATIO = 0.2
EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


class TrainingService:
    """Service for training models with new data."""
    
    def __init__(self, model_registry: ModelRegistry):
        """
        Initialize training service.
        
        Args:
            model_registry: Model registry instance
        """
        self.model_registry = model_registry
        self.feature_engineer = FeatureEngineer()
    
    def train_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train models with new data.
        
        Args:
            data: DataFrame with stock data (must have required columns)
            
        Returns:
            Dictionary with training results and model version
        """
        logger.info("Starting model training with new data...")
        
        try:
            # Engineer features
            logger.info("Engineering features...")
            df_features = self.feature_engineer.create_features(data.copy())
            
            # Create target
            if "Target_Close" not in df_features.columns:
                if "Close Price" in df_features.columns:
                    df_features["Target_Close"] = df_features["Close Price"].shift(-1)
                else:
                    raise ValueError("Data must contain 'Close Price' column")
            
            df_features = df_features.dropna(subset=["Target_Close"])
            
            if len(df_features) < 100:
                raise ValueError(f"Insufficient data: need at least 100 rows, got {len(df_features)}")
            
            # Select features
            target_col = df_features["Target_Close"].copy()
            df_features = self.feature_engineer.select_features(df_features)
            df_features["Target_Close"] = target_col
            df_features = df_features.dropna(subset=["Target_Close"])
            
            # Prepare train/test split
            feature_cols = [c for c in df_features.columns if c != "Target_Close"]
            
            split_idx = int(len(df_features) * (1 - TEST_RATIO))
            train_df = df_features.iloc[:split_idx]
            test_df = df_features.iloc[split_idx:]
            
            # Further split train into train/val
            val_split_idx = int(len(train_df) * 0.8)
            train_final = train_df.iloc[:val_split_idx]
            val_df = train_df.iloc[val_split_idx:]
            
            X_train = train_final[feature_cols].values.astype(float)
            y_train = train_final["Target_Close"].values.astype(float)
            X_val = val_df[feature_cols].values.astype(float)
            y_val = val_df["Target_Close"].values.astype(float)
            X_test = test_df[feature_cols].values.astype(float)
            y_test = test_df["Target_Close"].values.astype(float)
            
            logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            # Train models
            results = {}
            
            # Train LightGBM
            logger.info("Training LightGBM...")
            lgb_model, lgb_val_pred, lgb_val_rmse, lgb_val_mae, lgb_val_r2 = self._train_lightgbm(
                X_train, y_train, X_val, y_val, feature_cols
            )
            lgb_test_pred = lgb_model.predict(X_test)
            lgb_test_rmse = np.sqrt(mean_squared_error(y_test, lgb_test_pred))
            lgb_test_mae = mean_absolute_error(y_test, lgb_test_pred)
            lgb_test_r2 = r2_score(y_test, lgb_test_pred)
            
            results["lightgbm"] = {
                "val_rmse": float(lgb_val_rmse),
                "val_mae": float(lgb_val_mae),
                "val_r2": float(lgb_val_r2),
                "test_rmse": float(lgb_test_rmse),
                "test_mae": float(lgb_test_mae),
                "test_r2": float(lgb_test_r2)
            }
            
            # Train XGBoost
            logger.info("Training XGBoost...")
            xgb_model, xgb_val_pred, xgb_val_rmse, xgb_val_mae, xgb_val_r2 = self._train_xgboost(
                X_train, y_train, X_val, y_val, feature_cols
            )
            xgb_test_pred = xgb_model.predict(xgb.DMatrix(X_test, feature_names=feature_cols))
            xgb_test_rmse = np.sqrt(mean_squared_error(y_test, xgb_test_pred))
            xgb_test_mae = mean_absolute_error(y_test, xgb_test_pred)
            xgb_test_r2 = r2_score(y_test, xgb_test_pred)
            
            results["xgboost"] = {
                "val_rmse": float(xgb_val_rmse),
                "val_mae": float(xgb_val_mae),
                "val_r2": float(xgb_val_r2),
                "test_rmse": float(xgb_test_rmse),
                "test_mae": float(xgb_test_mae),
                "test_r2": float(xgb_test_r2)
            }
            
            # Train LSTM
            logger.info("Training LSTM...")
            lstm_model, lstm_scaler, lstm_val_pred, lstm_val_rmse, lstm_val_mae, lstm_val_r2 = self._train_lstm(
                X_train, y_train, X_val, y_val, feature_cols
            )
            X_test_scaled = lstm_scaler.transform(X_test)
            lstm_model.eval()
            with torch.no_grad():
                lstm_test_pred = lstm_model(torch.FloatTensor(X_test_scaled).to(DEVICE)).cpu().numpy().flatten()
            lstm_test_rmse = np.sqrt(mean_squared_error(y_test, lstm_test_pred))
            lstm_test_mae = mean_absolute_error(y_test, lstm_test_pred)
            lstm_test_r2 = r2_score(y_test, lstm_test_pred)
            
            results["lstm"] = {
                "val_rmse": float(lstm_val_rmse),
                "val_mae": float(lstm_val_mae),
                "val_r2": float(lstm_val_r2),
                "test_rmse": float(lstm_test_rmse),
                "test_mae": float(lstm_test_mae),
                "test_r2": float(lstm_test_r2)
            }
            
            # Save models
            version = self._save_models(lgb_model, xgb_model, lstm_model, lstm_scaler, feature_cols)
            
            # Register models
            version_dir = settings.MODELS_DIR / version
            self.model_registry.register_model(version, "lightgbm", version_dir / "lightgbm_model.txt", {
                "feature_names": feature_cols,
                "test_rmse": float(lgb_test_rmse),
                "test_mae": float(lgb_test_mae),
                "test_r2": float(lgb_test_r2)
            })
            self.model_registry.register_model(version, "xgboost", version_dir / "xgboost_model.json", {
                "feature_names": feature_cols,
                "test_rmse": float(xgb_test_rmse),
                "test_mae": float(xgb_test_mae),
                "test_r2": float(xgb_test_r2)
            })
            self.model_registry.register_model(version, "lstm", version_dir / "lstm", {
                "input_dim": len(feature_cols),
                "hidden_dim": 128,
                "test_rmse": float(lstm_test_rmse),
                "test_mae": float(lstm_test_mae),
                "test_r2": float(lstm_test_r2)
            })
            
            # Set as active version
            self.model_registry.set_active_version(version)
            
            # Reload models
            self.model_registry.load_models(version)
            
            results["version"] = version
            results["feature_names"] = feature_cols
            results["num_features"] = len(feature_cols)
            results["train_size"] = len(X_train)
            results["val_size"] = len(X_val)
            results["test_size"] = len(X_test)
            
            logger.info(f"Training completed successfully. Model version: {version}")
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val, feature_names):
        """Train LightGBM model."""
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_names)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.01,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': SEED
        }
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
        )
        
        val_pred = model.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_mae = mean_absolute_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        return model, val_pred, val_rmse, val_mae, val_r2
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val, feature_names):
        """Train XGBoost model."""
        train_data = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        val_data = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': SEED
        }
        
        model = xgb.train(
            params,
            train_data,
            num_boost_round=1000,
            evals=[(val_data, 'validation')],
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        val_pred = model.predict(val_data)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_mae = mean_absolute_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        return model, val_pred, val_rmse, val_mae, val_r2
    
    def _train_lstm(self, X_train, y_train, X_val, y_val, feature_names):
        """Train LSTM model."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = LSTMModel(
            input_dim=X_train_scaled.shape[1],
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
            use_attention=True,
            bidirectional=True
        ).to(DEVICE)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(train_loader.dataset)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
            
            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= 30:
                break
        
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val_scaled).to(DEVICE)).cpu().numpy().flatten()
        
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_mae = mean_absolute_error(y_val, val_pred)
        val_r2 = r2_score(y_val, val_pred)
        
        return model, scaler, val_pred, val_rmse, val_mae, val_r2
    
    def _save_models(self, lgb_model, xgb_model, lstm_model, lstm_scaler, feature_names):
        """Save all trained models."""
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = settings.MODELS_DIR / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM
        lgb_path = version_dir / "lightgbm_model.txt"
        lgb_model.save_model(str(lgb_path))
        
        lgb_metadata = {
            "version": version,
            "model_type": "lightgbm",
            "feature_names": feature_names,
            "trained_at": datetime.now().isoformat()
        }
        with open(version_dir / "lightgbm_metadata.pkl", "wb") as f:
            pickle.dump(lgb_metadata, f)
        
        # Save XGBoost
        xgb_path = version_dir / "xgboost_model.json"
        xgb_model.save_model(str(xgb_path))
        
        xgb_metadata = {
            "version": version,
            "model_type": "xgboost",
            "feature_names": feature_names,
            "trained_at": datetime.now().isoformat()
        }
        with open(version_dir / "xgboost_metadata.pkl", "wb") as f:
            pickle.dump(xgb_metadata, f)
        
        # Save LSTM
        lstm_path = version_dir / "lstm"
        lstm_path.mkdir(parents=True, exist_ok=True)
        
        torch.save(lstm_model.state_dict(), lstm_path / "model.pth")
        with open(lstm_path / "scaler.pkl", "wb") as f:
            pickle.dump(lstm_scaler, f)
        
        lstm_metadata = {
            "version": version,
            "model_type": "lstm",
            "input_dim": len(feature_names),
            "hidden_dim": 128,
            "num_layers": 2,
            "use_attention": True,
            "bidirectional": True,
            "trained_at": datetime.now().isoformat()
        }
        with open(lstm_path / "metadata.pkl", "wb") as f:
            pickle.dump(lstm_metadata, f)
        
        return version
