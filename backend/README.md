# Stock Prediction Backend API

## Architecture Overview

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration management
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py           # API route handlers
│   │   └── schemas.py          # Pydantic models for request/response
│   ├── services/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py  # Feature pipeline
│   │   ├── model_registry.py       # Model versioning & loading
│   │   ├── prediction_service.py   # Ensemble prediction logic
│   │   ├── backtesting.py          # Walk-forward backtesting
│   │   └── explainability.py       # SHAP-based explanations
│   ├── models/
│   │   ├── __init__.py
│   │   ├── xgboost_model.py    # XGBoost wrapper
│   │   └── neural_network.py   # PyTorch NN wrapper
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py      # Data loading utilities
│       └── metrics.py          # Performance metrics
├── models/                      # Trained model artifacts
│   ├── xgboost/
│   └── neural_network/
├── data/                        # Raw and processed data
├── requirements.txt
└── .env.example
```

## Design Decisions

1. **Service Layer Pattern**: Separates business logic from API routes for testability and reusability
2. **Model Registry**: Centralized model versioning for A/B testing and rollback capabilities
3. **Feature Store**: Versioned feature engineering pipeline for reproducibility
4. **Async Support**: FastAPI async endpoints for I/O-bound operations (data loading, model inference)
5. **Caching Strategy**: Redis for prediction caching (same input = cached result)

## API Endpoints

- `POST /api/v1/predict` - Get stock price prediction
- `POST /api/v1/backtest` - Run walk-forward backtest
- `POST /api/v1/explain` - Get SHAP-based feature importance
- `GET /api/v1/models` - List available models and versions
- `GET /api/v1/health` - Health check

## Model Ensemble Strategy

- **XGBoost**: Primary model for capturing non-linear patterns
- **Neural Network**: Residual learner to capture XGBoost's missed patterns
- **Dynamic Weighting**: Based on recent validation performance
- **Confidence Estimation**: Prediction intervals using quantile regression

