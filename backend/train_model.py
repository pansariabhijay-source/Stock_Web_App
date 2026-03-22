"""
Production training script for stock prediction models.
Converts notebook logic into a reusable training pipeline.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import json
import logging

from app.services.feature_engineering import FeatureEngineer
from app.services.model_registry import ModelRegistry
from app.utils.data_loader import DataLoader as DataLoaderUtil
from app.models.neural_network import ResidualNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SEED = 42
TEST_RATIO = 0.2
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.0003
MODELS_DIR = Path("models")
DATA_DIR = Path("data")

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Set random seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_preprocess_data(data_path: Path) -> pd.DataFrame:
    """Load and preprocess raw stock data."""
    logger.info(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    
    # Parse date
    if 'Date' not in df.columns:
        df.columns.values[0] = 'Date'
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.set_index('Date')
    
    # Ensure numeric types
    numeric_cols = [
        "Open Price", "High Price", "Low Price", "Last Price",
        "Close Price", "Average Price", "Total Traded Quantity",
        "Turnover", "No. of Trades", "Deliverable Qty"
    ]
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values
    price_cols = [c for c in numeric_cols if c in df.columns and "Price" in c]
    if price_cols:
        df[price_cols] = df[price_cols].ffill().bfill()
    
    # Fill remaining with median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features using FeatureEngineer."""
    logger.info("Engineering features...")
    
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.create_features(df)
    
    # Create target
    df_features["Target_Close"] = df_features["Close Price"].shift(-1)
    df_features = df_features.dropna(subset=["Target_Close"])
    
    # Check for duplicate indices and fix if needed
    if df_features.index.duplicated().any():
        logger.warning("Duplicate indices detected, using integer index")
        # Reset to integer index to avoid duplicate index issues
        df_features = df_features.reset_index(drop=False)
        df_features = df_features.set_index(pd.RangeIndex(len(df_features)))
    
    # Select recommended features (preserve Target_Close separately)
    target_col = df_features["Target_Close"].copy()
    df_features = feature_engineer.select_features(df_features)
    # Re-add Target_Close after feature selection (alignment by index)
    df_features["Target_Close"] = target_col
    df_features = df_features.dropna(subset=["Target_Close"])
    
    logger.info(f"Feature engineering complete. Shape: {df_features.shape}")
    return df_features


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model."""
    logger.info("Training XGBoost model...")
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.003,
        "max_depth": 6,
        "min_child_weight": 5,
        "gamma": 0.1,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "tree_method": "hist",
        "seed": SEED,
        "n_jobs": -1
    }
    
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=500,
        evals=[(dtest, "test")],
        early_stopping_rounds=50,
        verbose_eval=50
    )
    
    # Evaluate
    train_pred = model.predict(dtrain)
    test_pred = model.predict(dtest)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mape = mean_absolute_percentage_error(y_test, test_pred)
    
    logger.info(f"XGBoost - Train RMSE: {train_rmse:.6f}, Test RMSE: {test_rmse:.6f}, MAPE: {test_mape:.6f}")
    
    return model, test_pred, test_rmse, test_mape


def train_neural_network(X_train, y_train, X_test, y_test, xgb_residuals_train, xgb_residuals_test):
    """Train neural network on XGBoost residuals."""
    logger.info("Training Neural Network on residuals...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create dataset
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(xgb_residuals_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = ResidualNN(X_train_scaled.shape[1], hidden_dims=[64, 32]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader.dataset):.6f}")
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        resid_pred_test = model(torch.FloatTensor(X_test_scaled).to(DEVICE)).cpu().numpy()
    
    test_rmse = np.sqrt(mean_squared_error(xgb_residuals_test, resid_pred_test))
    logger.info(f"Neural Network - Test RMSE on residuals: {test_rmse:.6f}")
    
    return model, scaler, resid_pred_test


def save_models(xgb_model, nn_model, scaler, feature_names, version: str = None):
    """Save trained models with metadata."""
    if version is None:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Saving models version {version}")
    
    # Create version directory
    version_dir = MODELS_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Save Neural Network
    nn_path = version_dir / "neural_network"
    nn_path.mkdir(parents=True, exist_ok=True)
    
    torch.save(nn_model.state_dict(), nn_path / "model.pth")
    
    with open(nn_path / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    nn_metadata = {
        "version": version,
        "model_type": "neural_network",
        "input_dim": len(feature_names),
        "hidden_dims": [64, 32],
        "trained_at": datetime.now().isoformat()
    }
    with open(nn_path / "metadata.pkl", "wb") as f:
        pickle.dump(nn_metadata, f)
    
    logger.info(f"Models saved to {version_dir}")
    return version


def main():
    """Main training pipeline."""
    logger.info("Starting model training pipeline...")
    
    # Load data
    data_path = DATA_DIR / "RILO - Copy.csv"
    if not data_path.exists():
        # Try root directory
        data_path = Path("RILO - Copy.csv")
        if not data_path.exists():
            raise FileNotFoundError("Data file not found. Please provide RILO - Copy.csv")
    
    df = load_and_preprocess_data(data_path)
    
    # Engineer features
    df_features = engineer_features(df)
    
    # Prepare train/test split
    feature_cols = [c for c in df_features.columns if c != "Target_Close"]
    target_col = "Target_Close"
    
    split_idx = int(len(df_features) * (1 - TEST_RATIO))
    train_df = df_features.iloc[:split_idx]
    test_df = df_features.iloc[split_idx:]
    
    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df[target_col].values.astype(float)
    X_test = test_df[feature_cols].values.astype(float)
    y_test = test_df[target_col].values.astype(float)
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Train XGBoost
    xgb_model, xgb_test_pred, xgb_rmse, xgb_mape = train_xgboost(X_train, y_train, X_test, y_test)
    
    # Calculate residuals
    xgb_residuals_train = y_train - xgb_model.predict(xgb.DMatrix(X_train, label=y_train))
    xgb_residuals_test = y_test - xgb_test_pred
    
    # Train Neural Network
    nn_model, scaler, nn_residual_pred = train_neural_network(
        X_train, y_train, X_test, y_test, xgb_residuals_train, xgb_residuals_test
    )
    
    # Ensemble prediction
    ensemble_pred = xgb_test_pred + nn_residual_pred
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    ensemble_mape = mean_absolute_percentage_error(y_test, ensemble_pred)
    
    improvement = 100 * (xgb_rmse - ensemble_rmse) / xgb_rmse
    
    logger.info("=" * 50)
    logger.info("FINAL RESULTS")
    logger.info("=" * 50)
    logger.info(f"XGBoost RMSE: {xgb_rmse:.6f}, MAPE: {xgb_mape:.6f}")
    logger.info(f"Ensemble RMSE: {ensemble_rmse:.6f}, MAPE: {ensemble_mape:.6f}")
    logger.info(f"Improvement: {improvement:.2f}%")
    logger.info("=" * 50)
    
    # Save models
    version = save_models(xgb_model, nn_model, scaler, feature_cols)
    
    # Register models in registry
    logger.info(f"Registering models version {version} in registry...")
    registry = ModelRegistry()
    
    version_dir = MODELS_DIR / version
    xgb_path = version_dir / "xgboost_model.json"
    nn_path = version_dir / "neural_network"
    
    # Register models with metadata
    xgb_metadata = {
        "feature_names": feature_cols,
        "test_rmse": float(xgb_rmse),
        "test_mape": float(xgb_mape)
    }
    nn_metadata = {
        "input_dim": len(feature_cols),
        "hidden_dims": [64, 32]
    }
    
    registry.register_model(version, "xgboost", xgb_path, xgb_metadata)
    registry.register_model(version, "neural_network", nn_path, nn_metadata)
    
    # Set as active version
    registry.set_active_version(version)
    logger.info(f"Models version {version} registered and set as active")
    
    # Save final features dataset
    df_features.to_csv(DATA_DIR / "FINAL_FEATURES_OUT.csv")
    logger.info("Saved processed features dataset")
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()

