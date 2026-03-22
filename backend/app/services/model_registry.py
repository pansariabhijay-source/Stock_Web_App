"""
Model Registry for versioning and managing ML models.
Supports model loading, versioning, and A/B testing.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from app.models.xgboost_model import XGBoostModel
from app.models.lightgbm_model import LightGBMModel
from app.models.lstm_model import LSTMStockModel
from app.models.neural_network import NeuralNetworkModel
from app.config import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Centralized model registry for versioning and management.
    
    Tracks:
    - Model versions
    - Performance metrics
    - Deployment status
    - Feature versions
    Supports: XGBoost, LightGBM, LSTM, Neural Network
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = registry_path or settings.MODELS_DIR / "registry.json"
        self.registry: Dict[str, Any] = self._load_registry()
        self.xgb_model: Optional[XGBoostModel] = None
        self.lgb_model: Optional[LightGBMModel] = None
        self.lstm_model: Optional[LSTMStockModel] = None
        self.nn_model: Optional[NeuralNetworkModel] = None
        self.current_version: Optional[str] = None
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from file or create new."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}. Creating new registry.")
        
        return {
            "models": {},
            "active_version": None,
            "versions": []
        }
    
    def _save_registry(self) -> None:
        """Save registry to file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(
        self,
        version: str,
        model_type: str,
        model_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a new model version.
        
        Args:
            version: Version identifier (e.g., "v1.0.0")
            model_type: "xgboost" or "neural_network"
            model_path: Path to model file
            metadata: Additional metadata (metrics, feature_version, etc.)
        """
        if version not in self.registry["models"]:
            self.registry["models"][version] = {}
        
        # Convert to relative path if it's within MODELS_DIR
        path = Path(model_path)
        try:
            relative_path = path.relative_to(settings.MODELS_DIR)
            # If the relative path starts with "models/", remove it (shouldn't happen but handle it)
            if str(relative_path).startswith("models/"):
                relative_path = Path(*Path(relative_path).parts[1:])
        except ValueError:
            # Path is not relative to MODELS_DIR, try to make relative
            if path.is_absolute():
                # Try to extract version directory name
                if version in str(path):
                    # Extract the part after models/
                    parts = path.parts
                    try:
                        models_idx = parts.index('models')
                        relative_path = Path(*parts[models_idx + 1:])
                        # Remove another "models/" if it exists in the relative path
                        if str(relative_path).startswith("models/"):
                            relative_path = Path(*Path(relative_path).parts[1:])
                    except (ValueError, IndexError):
                        relative_path = Path(f"{version}/{path.name}" if model_type == "xgboost" else f"{version}/neural_network")
                else:
                    relative_path = path
            else:
                # If it's a relative path that starts with "models/", remove that prefix
                if str(path).startswith("models/"):
                    relative_path = Path(*path.parts[1:])
                else:
                    relative_path = path
        
        model_info = {
            "version": version,
            "model_type": model_type,
            "model_path": str(relative_path),
            "registered_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.registry["models"][version][model_type] = model_info
        
        if version not in self.registry["versions"]:
            self.registry["versions"].append(version)
        
        self._save_registry()
        logger.info(f"Registered {model_type} model version {version}")
    
    def load_models(self, version: Optional[str] = None) -> None:
        """
        Load models for a specific version (or active version).
        
        Args:
            version: Model version to load. If None, loads active version.
        """
        version = version or self.registry.get("active_version")
        if version is None:
            # Try to load from config paths
            if settings.XGBOOST_MODEL_PATH and settings.NN_MODEL_PATH:
                logger.info("Loading models from config paths")
                self._load_from_paths(
                    Path(settings.XGBOOST_MODEL_PATH),
                    Path(settings.NN_MODEL_PATH)
                )
                return
            
            # Auto-detect latest model version if registry is empty
            logger.info("No registry found, auto-detecting latest model version...")
            version = self._find_latest_version()
            if version:
                logger.info(f"Found model version: {version}")
                # Auto-register it
                self._auto_register_version(version)
            else:
                raise ValueError("No model version specified and no models found in models directory")
        
        if version not in self.registry["models"]:
            # Try to auto-register if model files exist
            if self._try_auto_register_version(version):
                logger.info(f"Auto-registered model version {version}")
            else:
                raise ValueError(f"Model version {version} not found in registry and could not be auto-registered")
        
        model_info = self.registry["models"][version]
        
        # Helper function to resolve model paths
        def resolve_path(path_str: str) -> Path:
            path = Path(path_str)
            if str(path).startswith("models/"):
                path = Path(*path.parts[1:])
            if not path.is_absolute():
                path = settings.MODELS_DIR / path
            return path
        
        # Track loading success/failure
        loaded_models = []
        failed_models = []
        
        # Load LightGBM (preferred)
        if "lightgbm" in model_info:
            try:
                lgb_path = resolve_path(model_info["lightgbm"]["model_path"])
                if not lgb_path.exists():
                    raise FileNotFoundError(f"LightGBM model file not found: {lgb_path}")
                self.lgb_model = LightGBMModel(lgb_path)
                self.lgb_model.load()
                loaded_models.append("lightgbm")
                logger.info(f"Loaded LightGBM model version {version} from {lgb_path}")
            except Exception as e:
                failed_models.append(("lightgbm", str(e)))
                logger.error(f"Failed to load LightGBM model: {e}", exc_info=True)
        
        # Load XGBoost
        if "xgboost" in model_info:
            try:
                xgb_path = resolve_path(model_info["xgboost"]["model_path"])
                if not xgb_path.exists():
                    raise FileNotFoundError(f"XGBoost model file not found: {xgb_path}")
                self.xgb_model = XGBoostModel(xgb_path)
                self.xgb_model.load()
                loaded_models.append("xgboost")
                logger.info(f"Loaded XGBoost model version {version} from {xgb_path}")
            except Exception as e:
                failed_models.append(("xgboost", str(e)))
                logger.error(f"Failed to load XGBoost model: {e}", exc_info=True)
        
        # Load LSTM
        if "lstm" in model_info:
            try:
                lstm_path = resolve_path(model_info["lstm"]["model_path"])
                if not lstm_path.exists():
                    raise FileNotFoundError(f"LSTM model path not found: {lstm_path}")
                self.lstm_model = LSTMStockModel(lstm_path)
                self.lstm_model.load()
                loaded_models.append("lstm")
                logger.info(f"Loaded LSTM model version {version} from {lstm_path}")
            except Exception as e:
                failed_models.append(("lstm", str(e)))
                logger.error(f"Failed to load LSTM model: {e}", exc_info=True)
        
        # Load Neural Network (legacy support)
        if "neural_network" in model_info:
            try:
                nn_path = resolve_path(model_info["neural_network"]["model_path"])
                if not nn_path.exists():
                    raise FileNotFoundError(f"Neural Network model path not found: {nn_path}")
                self.nn_model = NeuralNetworkModel(nn_path)
                self.nn_model.load()
                loaded_models.append("neural_network")
                logger.info(f"Loaded Neural Network model version {version} from {nn_path}")
            except Exception as e:
                failed_models.append(("neural_network", str(e)))
                logger.error(f"Failed to load Neural Network model: {e}", exc_info=True)
        
        # Set current version if at least one model loaded
        if loaded_models:
            self.current_version = version
            logger.info(f"Successfully loaded {len(loaded_models)} model(s): {', '.join(loaded_models)}")
            if failed_models:
                logger.warning(f"Failed to load {len(failed_models)} model(s): {', '.join([f'{m[0]}: {m[1]}' for m in failed_models])}")
        else:
            # No models loaded successfully
            error_msg = f"Failed to load any models for version {version}"
            if failed_models:
                error_details = "; ".join([f"{m[0]}: {m[1]}" for m in failed_models])
                error_msg += f". Errors: {error_details}"
            raise ValueError(error_msg)
    
    def _load_from_paths(self, xgb_path: Path, nn_path: Path) -> None:
        """Load models directly from paths (fallback)."""
        self.xgb_model = XGBoostModel(xgb_path)
        self.xgb_model.load()
        
        self.nn_model = NeuralNetworkModel(nn_path)
        self.nn_model.load()
        
        self.current_version = "default"
    
    def set_active_version(self, version: str, allow_unregistered: bool = False) -> None:
        """Set active model version."""
        if not allow_unregistered and version not in self.registry["models"]:
            raise ValueError(f"Version {version} not found")
        
        self.registry["active_version"] = version
        self._save_registry()
        logger.info(f"Set active version to {version}")
    
    def get_model_info(self, version: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a model version."""
        version = version or self.current_version or self.registry.get("active_version")
        if version is None:
            return {}
        
        return self.registry["models"].get(version, {})
    
    def list_versions(self) -> List[str]:
        """List all registered model versions."""
        return self.registry["versions"].copy()
    
    def get_loading_status(self) -> Dict[str, Any]:
        """
        Get diagnostic information about model loading status.
        
        Returns:
            Dictionary with model loading status, registered models, and file existence checks
        """
        status = {
            "current_version": self.current_version,
            "active_version": self.registry.get("active_version"),
            "models_loaded": {
                "lightgbm": self.lgb_model is not None,
                "xgboost": self.xgb_model is not None,
                "lstm": self.lstm_model is not None,
                "neural_network": self.nn_model is not None
            },
            "registered_versions": self.registry["versions"],
            "registry_path": str(self.registry_path),
            "registry_exists": self.registry_path.exists()
        }
        
        # Check file existence for active version
        active_version = self.registry.get("active_version")
        if active_version and active_version in self.registry["models"]:
            model_info = self.registry["models"][active_version]
            file_checks = {}
            
            def resolve_path(path_str: str) -> Path:
                path = Path(path_str)
                if str(path).startswith("models/"):
                    path = Path(*path.parts[1:])
                if not path.is_absolute():
                    path = settings.MODELS_DIR / path
                return path
            
            for model_type in ["lightgbm", "xgboost", "lstm", "neural_network"]:
                if model_type in model_info:
                    model_path_str = model_info[model_type]["model_path"]
                    resolved_path = resolve_path(model_path_str)
                    file_checks[model_type] = {
                        "registered_path": model_path_str,
                        "resolved_path": str(resolved_path),
                        "exists": resolved_path.exists()
                    }
            
            status["active_version_file_checks"] = file_checks
        
        return status
    
    def _find_latest_version(self) -> Optional[str]:
        """Find the latest model version by scanning models directory."""
        if not settings.MODELS_DIR.exists():
            return None
        
        # Look for version directories (format: YYYYMMDD_HHMMSS)
        version_dirs = [d for d in settings.MODELS_DIR.iterdir() 
                       if d.is_dir() and not d.name.startswith('.')]
        
        if not version_dirs:
            return None
        
        # Sort by directory name (which should be timestamp-based)
        version_dirs.sort(reverse=True)
        
        # Check if the latest directory has at least one model
        latest_dir = version_dirs[0]
        lgb_path = latest_dir / "lightgbm_model.txt"
        xgb_path = latest_dir / "xgboost_model.json"
        lstm_path = latest_dir / "lstm" / "model.pth"
        nn_path = latest_dir / "neural_network" / "model.pth"
        
        # Prefer LightGBM, then XGBoost, then LSTM, then NN
        if lgb_path.exists():
            return latest_dir.name
        elif xgb_path.exists():
            return latest_dir.name
        elif lstm_path.exists():
            return latest_dir.name
        elif nn_path.exists():
            return latest_dir.name
        
        return None
    
    def _auto_register_version(self, version: str) -> None:
        """Auto-register a model version by detecting files."""
        version_dir = settings.MODELS_DIR / version
        
        # Register all available models
        lgb_path = version_dir / "lightgbm_model.txt"
        xgb_path = version_dir / "xgboost_model.json"
        lstm_path = version_dir / "lstm"
        nn_path = version_dir / "neural_network"
        
        if lgb_path.exists():
            self.register_model(version, "lightgbm", lgb_path, {})
        
        if xgb_path.exists():
            self.register_model(version, "xgboost", xgb_path, {})
        
        if lstm_path.exists() and (lstm_path / "model.pth").exists():
            self.register_model(version, "lstm", lstm_path, {})
        
        if nn_path.exists() and (nn_path / "model.pth").exists():
            self.register_model(version, "neural_network", nn_path, {})
        
        # Set as active version (after registration)
        if version in self.registry["models"]:
            self.set_active_version(version)
    
    def _try_auto_register_version(self, version: str) -> bool:
        """Try to auto-register a version if files exist."""
        version_dir = settings.MODELS_DIR / version
        
        if not version_dir.exists():
            return False
        
        # Check if at least one model exists
        lgb_path = version_dir / "lightgbm_model.txt"
        xgb_path = version_dir / "xgboost_model.json"
        lstm_path = version_dir / "lstm" / "model.pth"
        nn_path = version_dir / "neural_network" / "model.pth"
        
        has_any_model = (
            lgb_path.exists() or
            xgb_path.exists() or
            lstm_path.exists() or
            nn_path.exists()
        )
        
        if not has_any_model:
            return False
        
        self._auto_register_version(version)
        return True

