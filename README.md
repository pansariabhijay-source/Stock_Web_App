# Stock Prediction Web Application

Production-grade ML system for stock price prediction with ensemble models (XGBoost + Neural Network).

## 🏗️ Architecture

```
Stock_Prediction_Model/
├── backend/              # FastAPI backend
│   ├── app/
│   │   ├── api/         # REST API routes
│   │   ├── services/    # Business logic
│   │   ├── models/      # ML model wrappers
│   │   └── utils/       # Utilities
│   ├── models/          # Trained model artifacts
│   ├── data/            # Processed data
│   └── train_model.py   # Training pipeline
│
├── frontend/            # Streamlit frontend
│   ├── app.py          # Main application
│   ├── utils/          # Frontend utilities
│   └── pages/          # Page components
│
└── [data files]        # Original data files
```

## 🚀 Quick Start

### Backend Setup

1. **Install dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

2. **Train models:**
```bash
# Copy your data file to backend/data/
cp "RILO - Copy.csv" backend/data/

# Train models
python train_model.py
```

3. **Configure environment:**
```bash
# Create .env file (optional)
cp .env.example .env
# Edit .env with your settings
```

4. **Run API:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/api/v1/health`

### Frontend Setup

1. **Install dependencies:**
```bash
cd frontend
pip install -r requirements.txt
```

2. **Configure API URL:**
```bash
# Create secrets file
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml and set API_URL
```

3. **Run Streamlit:**
```bash
streamlit run app.py
```

Frontend will be available at `http://localhost:8501`

## 📋 Features

### Backend (FastAPI)
- ✅ Feature engineering pipeline (technical indicators, lag features)
- ✅ Model registry with versioning
- ✅ Ensemble prediction (XGBoost + Neural Network)
- ✅ Walk-forward backtesting
- ✅ SHAP-based explainability
- ✅ Confidence intervals
- ✅ Performance metrics tracking

### Frontend (Streamlit)
- ✅ Interactive prediction interface
- ✅ Backtesting visualization (equity curve, drawdown)
- ✅ SHAP explainability charts
- ✅ Model comparison
- ✅ Real-time predictions with confidence bands

## 🔧 API Endpoints

- `POST /api/v1/predict` - Get stock price prediction
- `POST /api/v1/backtest` - Run walk-forward backtest
- `POST /api/v1/explain` - Get SHAP-based explanation
- `GET /api/v1/models` - List available models
- `GET /api/v1/health` - Health check

## 📊 Model Architecture

**Ensemble Approach:**
1. **XGBoost**: Primary model for capturing non-linear patterns
2. **Neural Network**: Residual learner to capture XGBoost's missed patterns
3. **Dynamic Weighting**: Based on recent validation performance

**Features:**
- Technical indicators (RSI, MACD, Bollinger Bands)
- Lag features (1, 2, 3, 5, 7 days)
- Rolling statistics (MA, volatility, momentum)
- Volume-based features

## 🎯 Production Considerations

### Model Deployment
- Models are versioned and stored in `backend/models/`
- Model registry tracks versions and metadata
- Easy rollback to previous versions

### Monitoring (To Implement)
- Model drift detection
- Prediction monitoring
- Performance tracking
- Alerting on degradation

### Scaling
- Backend can be deployed on Render/Railway
- Frontend can be deployed on Streamlit Cloud
- Add Redis caching for predictions
- Use PostgreSQL for model metadata

### Improvements
1. **Model Retraining Pipeline**: Automated retraining on schedule
2. **A/B Testing**: Compare model versions
3. **Feature Store**: Centralized feature management
4. **Data Pipeline**: Automated data ingestion
5. **Monitoring Dashboard**: Real-time metrics

## 📝 Notes

- The base model from `Stock_prediction.ipynb` is preserved and improved
- All original data files remain in root directory
- Backend and frontend are separated for independent deployment
- Models are trained using the same logic as the notebook

## 🔐 Environment Variables

### Backend (.env)
```
API_TITLE=Stock Prediction API
DEBUG=False
XGBOOST_MODEL_PATH=models/v1/xgboost_model.json
NN_MODEL_PATH=models/v1/neural_network
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Frontend (.streamlit/secrets.toml)
```
API_URL=http://localhost:8000
```

## 📚 Documentation

- Backend architecture: `backend/README.md`
- Frontend architecture: `frontend/README.md`
- API documentation: `http://localhost:8000/docs` (when running)

## 🤝 Contributing

This is a production-grade system designed for:
- SDE/ML interviews
- Portfolio projects
- Learning production ML systems

## 📄 License

MIT License

