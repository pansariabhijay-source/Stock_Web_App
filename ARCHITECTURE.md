# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (Streamlit)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │Prediction│  │Backtest  │  │Explain   │  │Model Info│    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/REST
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  Backend (FastAPI)                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              API Routes Layer                         │   │
│  │  /predict  /backtest  /explain  /models  /health    │   │
│  └──────────────────┬───────────────────────────────────┘   │
│                     │                                         │
│  ┌──────────────────▼───────────────────────────────────┐   │
│  │            Service Layer                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │   │
│  │  │Prediction    │  │Backtesting   │  │Explain    │  │   │
│  │  │Service       │  │Engine        │  │Service    │  │   │
│  │  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘  │   │
│  └─────────┼─────────────────┼────────────────┼─────────┘   │
│            │                 │                 │             │
│  ┌─────────▼─────────────────▼─────────────────▼─────────┐ │
│  │            Model Registry                               │ │
│  │  ┌──────────────┐              ┌──────────────┐       │ │
│  │  │ XGBoost      │              │ Neural Net   │       │ │
│  │  │ Model        │              │ Model        │       │ │
│  │  └──────────────┘              └──────────────┘       │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │         Feature Engineering Pipeline                  │ │
│  │  Technical Indicators | Lag Features | Rolling Stats │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Frontend (Streamlit)

**Purpose**: User interface for predictions, backtesting, and explainability

**Components**:
- `app.py`: Main application with navigation
- `utils/api_client.py`: HTTP client for backend communication
- `utils/charts.py`: Plotly-based visualizations
- `utils/data_loader.py`: Data loading utilities

**Features**:
- Interactive prediction interface
- Backtesting visualization (equity curve, drawdown)
- SHAP explainability charts
- Model information display

### 2. Backend (FastAPI)

#### API Layer (`app/api/`)
- **routes.py**: REST endpoint handlers
- **schemas.py**: Pydantic models for request/response validation

**Endpoints**:
- `POST /api/v1/predict`: Get predictions with confidence intervals
- `POST /api/v1/backtest`: Run walk-forward backtest
- `POST /api/v1/explain`: Get SHAP-based explanations
- `GET /api/v1/models`: List registered models
- `GET /api/v1/health`: Health check

#### Service Layer (`app/services/`)

**PredictionService**:
- Combines XGBoost and Neural Network predictions
- Dynamic ensemble weighting
- Confidence interval estimation

**BacktestingEngine**:
- Walk-forward validation
- Time-series cross-validation
- Performance metrics calculation
- Equity curve and drawdown analysis

**ExplainabilityService**:
- SHAP value calculation
- Feature importance ranking
- Prediction-level explanations

**ModelRegistry**:
- Model versioning
- Model loading and management
- Metadata tracking

**FeatureEngineer**:
- Technical indicator calculation
- Lag feature creation
- Rolling statistics
- Feature selection

#### Model Layer (`app/models/`)

**XGBoostModel**:
- Wrapper for XGBoost Booster
- Prediction interface
- Feature importance extraction

**NeuralNetworkModel**:
- PyTorch model wrapper
- Residual prediction
- Feature scaling

### 3. Data Flow

#### Prediction Flow:
```
1. User inputs features → Frontend
2. Frontend → API POST /predict
3. API → PredictionService
4. PredictionService → ModelRegistry (load models)
5. Models → Individual predictions
6. PredictionService → Ensemble prediction
7. API → Response with confidence intervals
8. Frontend → Display results
```

#### Backtesting Flow:
```
1. User triggers backtest → Frontend
2. Frontend → API POST /backtest
3. API → BacktestingEngine
4. BacktestingEngine → Load data, split train/test
5. For each test period:
   - Engineer features
   - Make predictions
   - Calculate metrics
6. BacktestingEngine → Aggregate results
7. API → Response with metrics and charts
8. Frontend → Visualize results
```

## Design Decisions

### 1. Service Layer Pattern
**Why**: Separates business logic from API routes, making code:
- Testable (services can be unit tested)
- Reusable (services can be used by different endpoints)
- Maintainable (clear separation of concerns)

### 2. Model Registry
**Why**: Centralized model management enables:
- Versioning and rollback
- A/B testing
- Model comparison
- Metadata tracking

### 3. Feature Engineering Pipeline
**Why**: Versioned feature engineering ensures:
- Reproducibility
- Consistency between training and inference
- Easy feature updates

### 4. Ensemble Approach
**Why**: XGBoost + Neural Network:
- XGBoost captures non-linear patterns well
- Neural Network learns residuals (what XGBoost misses)
- Combined: Better generalization

### 5. Walk-Forward Backtesting
**Why**: Time-series cross-validation:
- Avoids look-ahead bias
- Realistic performance estimation
- Handles concept drift

### 6. SHAP Explainability
**Why**: Model interpretability:
- Understand feature contributions
- Debug predictions
- Build trust with users

## Scalability Considerations

### Current Architecture
- Single-instance FastAPI server
- In-memory model loading
- File-based model storage

### Production Enhancements
1. **Horizontal Scaling**:
   - Load balancer (nginx)
   - Multiple FastAPI instances
   - Shared model storage (S3, GCS)

2. **Caching**:
   - Redis for prediction caching
   - Feature caching
   - Model response caching

3. **Database**:
   - PostgreSQL for model metadata
   - Time-series DB for predictions
   - Feature store (Feast, Tecton)

4. **Monitoring**:
   - Prometheus metrics
   - Grafana dashboards
   - Model drift detection
   - Alerting

5. **Model Serving**:
   - MLflow model serving
   - TensorFlow Serving (for NN)
   - Seldon Core

## Security Considerations

1. **API Authentication**: Add JWT tokens
2. **Rate Limiting**: Prevent abuse
3. **Input Validation**: Pydantic schemas
4. **CORS**: Configure for production
5. **Secrets Management**: Environment variables

## Deployment Architecture

### Development
```
Local Machine
├── Backend: uvicorn (localhost:8000)
└── Frontend: streamlit (localhost:8501)
```

### Production (Recommended)
```
Render / Railway (Backend)
├── FastAPI application
├── Model artifacts (S3/GCS)
└── Redis cache

Streamlit Cloud (Frontend)
├── Streamlit app
└── API client → Backend URL
```

## Future Improvements

1. **Model Retraining Pipeline**:
   - Scheduled retraining
   - Automated model validation
   - Model promotion workflow

2. **Feature Store**:
   - Centralized feature management
   - Feature versioning
   - Online/offline feature serving

3. **A/B Testing Framework**:
   - Model version comparison
   - Traffic splitting
   - Performance tracking

4. **Real-time Predictions**:
   - WebSocket support
   - Streaming predictions
   - Real-time data ingestion

5. **Advanced Monitoring**:
   - Model drift detection
   - Prediction monitoring
   - Performance dashboards

