"""
Production-grade training script for stock prediction models.
Implements multiple SOTA models: LightGBM, XGBoost, LSTM with proper validation.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

from app.services.feature_engineering import FeatureEngineer
from app.services.model_registry import ModelRegistry
from app.utils.data_loader import DataLoader as DataLoaderUtil
from app.models.lstm_model import LSTMModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SEED = 42
TEST_RATIO = 0.2
EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MODELS_DIR = Path("models")
DATA_DIR = Path("data")

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_preprocess_data(data_path: Path) -> pd.DataFrame:
    """Load and preprocess raw stock data."""
    logger.info(f"Loading data from {data_path}")
    
    df = pd.read_csv(data_path, index_col=0)
    df.columns = [c.strip() for c in df.columns]
    
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
    
    # Select recommended features
    target_col = df_features["Target_Close"].copy()
    df_features = feature_engineer.select_features(df_features)
    df_features["Target_Close"] = target_col
    df_features = df_features.dropna(subset=["Target_Close"])
    
    logger.info(f"Feature engineering complete. Shape: {df_features.shape}")
    return df_features


def train_lightgbm(X_train, y_train, X_val, y_val, feature_names):
    """Train LightGBM model with optimized hyperparameters."""
    logger.info("Training LightGBM model...")
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_names)
    
    # Optimized hyperparameters for stock prediction
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': SEED,
        'verbosity': -1,
        'force_col_wise': True
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Evaluate
    train_pred = model.predict(X_train, num_iteration=model.best_iteration)
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    logger.info(f"LightGBM - Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    return model, val_pred, val_rmse, val_mae, val_r2


def train_xgboost(X_train, y_train, X_val, y_val, feature_names):
    """Train XGBoost model with optimized hyperparameters."""
    logger.info("Training XGBoost model...")
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    
    # Optimized hyperparameters
    xgb_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.01,
        "max_depth": 6,
        "min_child_weight": 5,
        "gamma": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "tree_method": "hist",
        "seed": SEED,
        "n_jobs": -1
    }
    
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    # Evaluate
    train_pred = model.predict(dtrain)
    val_pred = model.predict(dval)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    logger.info(f"XGBoost - Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    return model, val_pred, val_rmse, val_mae, val_r2


def train_lstm(X_train, y_train, X_val, y_val, feature_names):
    """Train LSTM model for time-series prediction."""
    logger.info("Training LSTM model...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create datasets
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
    
    # Initialize model
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
    
    # Training loop
    for epoch in range(EPOCHS):
        # Training
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
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        
        val_loss /= len(val_loader.dataset)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Log learning rate changes
        if new_lr != current_lr:
            logger.info(f"Learning rate reduced from {current_lr:.6f} to {new_lr:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {new_lr:.6f}")
        
        if patience_counter >= 30:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        val_pred = model(torch.FloatTensor(X_val_scaled).to(DEVICE)).cpu().numpy()
    
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_mae = mean_absolute_error(y_val, val_pred)
    val_r2 = r2_score(y_val, val_pred)
    
    logger.info(f"LSTM - Val RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R²: {val_r2:.4f}")
    
    return model, scaler, val_pred, val_rmse, val_mae, val_r2


def save_models(lgb_model, xgb_model, lstm_model, lstm_scaler, feature_names, version: str = None):
    """Save all trained models with metadata."""
    if version is None:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Saving models version {version}")
    
    # Create version directory
    version_dir = MODELS_DIR / version
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
    
    logger.info(f"Models saved to {version_dir}")
    return version


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("PRODUCTION-GRADE MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Load data
    data_path = DATA_DIR / "RILO - Copy.csv"
    if not data_path.exists():
        data_path = Path("RILO - Copy.csv")
        if not data_path.exists():
            raise FileNotFoundError("Data file not found. Please provide RILO - Copy.csv")
    
    df = load_and_preprocess_data(data_path)
    
    # Engineer features
    df_features = engineer_features(df)
    
    # Prepare train/test split (time-series aware)
    feature_cols = [c for c in df_features.columns if c != "Target_Close"]
    target_col = "Target_Close"
    
    split_idx = int(len(df_features) * (1 - TEST_RATIO))
    train_df = df_features.iloc[:split_idx]
    test_df = df_features.iloc[split_idx:]
    
    # Further split train into train/val
    val_split_idx = int(len(train_df) * 0.8)
    train_final = train_df.iloc[:val_split_idx]
    val_df = train_df.iloc[val_split_idx:]
    
    X_train = train_final[feature_cols].values.astype(float)
    y_train = train_final[target_col].values.astype(float)
    X_val = val_df[feature_cols].values.astype(float)
    y_val = val_df[target_col].values.astype(float)
    X_test = test_df[feature_cols].values.astype(float)
    y_test = test_df[target_col].values.astype(float)
    
    logger.info(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
    
    # Train models
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING MODELS")
    logger.info("=" * 60)
    
    # Train LightGBM
    lgb_model, lgb_val_pred, lgb_val_rmse, lgb_val_mae, lgb_val_r2 = train_lightgbm(
        X_train, y_train, X_val, y_val, feature_cols
    )
    
    # Train XGBoost
    xgb_model, xgb_val_pred, xgb_val_rmse, xgb_val_mae, xgb_val_r2 = train_xgboost(
        X_train, y_train, X_val, y_val, feature_cols
    )
    
    # Train LSTM
    lstm_model, lstm_scaler, lstm_val_pred, lstm_val_rmse, lstm_val_mae, lstm_val_r2 = train_lstm(
        X_train, y_train, X_val, y_val, feature_cols
    )
    
    # Evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 60)
    
    # LightGBM test predictions
    lgb_test_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    lgb_test_rmse = np.sqrt(mean_squared_error(y_test, lgb_test_pred))
    lgb_test_mae = mean_absolute_error(y_test, lgb_test_pred)
    lgb_test_r2 = r2_score(y_test, lgb_test_pred)
    
    # XGBoost test predictions
    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
    xgb_test_pred = xgb_model.predict(dtest)
    xgb_test_rmse = np.sqrt(mean_squared_error(y_test, xgb_test_pred))
    xgb_test_mae = mean_absolute_error(y_test, xgb_test_pred)
    xgb_test_r2 = r2_score(y_test, xgb_test_pred)
    
    # LSTM test predictions
    X_test_scaled = lstm_scaler.transform(X_test)
    lstm_model.eval()
    with torch.no_grad():
        lstm_test_pred = lstm_model(torch.FloatTensor(X_test_scaled).to(DEVICE)).cpu().numpy()
    lstm_test_rmse = np.sqrt(mean_squared_error(y_test, lstm_test_pred))
    lstm_test_mae = mean_absolute_error(y_test, lstm_test_pred)
    lstm_test_r2 = r2_score(y_test, lstm_test_pred)
    
    # Ensemble prediction (weighted average)
    ensemble_pred = (
        0.4 * lgb_test_pred + 
        0.4 * xgb_test_pred + 
        0.2 * lstm_test_pred.flatten()
    )
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"{'Model':<15} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    logger.info("-" * 60)
    logger.info(f"{'LightGBM':<15} {lgb_test_rmse:<12.4f} {lgb_test_mae:<12.4f} {lgb_test_r2:<12.4f}")
    logger.info(f"{'XGBoost':<15} {xgb_test_rmse:<12.4f} {xgb_test_mae:<12.4f} {xgb_test_r2:<12.4f}")
    logger.info(f"{'LSTM':<15} {lstm_test_rmse:<12.4f} {lstm_test_mae:<12.4f} {lstm_test_r2:<12.4f}")
    logger.info(f"{'Ensemble':<15} {ensemble_rmse:<12.4f} {ensemble_mae:<12.4f} {ensemble_r2:<12.4f}")
    logger.info("=" * 60)
    
    # Save models
    version = save_models(lgb_model, xgb_model, lstm_model, lstm_scaler, feature_cols)
    
    # Register models in registry
    logger.info(f"\nRegistering models version {version} in registry...")
    registry = ModelRegistry()
    
    version_dir = MODELS_DIR / version
    lgb_path = version_dir / "lightgbm_model.txt"
    xgb_path = version_dir / "xgboost_model.json"
    lstm_path = version_dir / "lstm"
    
    # Register with metadata including test metrics
    lgb_meta = {
        "feature_names": feature_cols,
        "test_rmse": float(lgb_test_rmse),
        "test_mae": float(lgb_test_mae),
        "test_r2": float(lgb_test_r2)
    }
    xgb_meta = {
        "feature_names": feature_cols,
        "test_rmse": float(xgb_test_rmse),
        "test_mae": float(xgb_test_mae),
        "test_r2": float(xgb_test_r2)
    }
    lstm_meta = {
        "input_dim": len(feature_cols),
        "hidden_dim": 128,
        "test_rmse": float(lstm_test_rmse),
        "test_mae": float(lstm_test_mae),
        "test_r2": float(lstm_test_r2)
    }
    
    registry.register_model(version, "lightgbm", lgb_path, lgb_meta)
    registry.register_model(version, "xgboost", xgb_path, xgb_meta)
    registry.register_model(version, "lstm", lstm_path, lstm_meta)
    
    # Set as active version
    registry.set_active_version(version)
    logger.info(f"Models version {version} registered and set as active")
    
    # Save final features dataset
    df_features.to_csv(DATA_DIR / "FINAL_FEATURES_OUT.csv")
    logger.info("Saved processed features dataset")
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

