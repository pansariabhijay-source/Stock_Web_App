# Quick Start Guide

Get your Stock Prediction application running in 5 minutes!

## Prerequisites

- Python 3.9 or higher
- pip package manager

## Step 1: Setup Project

```bash
# Organize files (if not already done)
python setup_project.py
```

## Step 2: Train Models

```bash
cd backend
pip install -r requirements.txt

# Train models (this will take a few minutes)
python train_model.py
```

This will:
- Load and preprocess your data
- Engineer features
- Train XGBoost model
- Train Neural Network
- Save models to `backend/models/`

## Step 3: Start Backend

```bash
# In backend directory
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/api/v1/health`

## Step 4: Start Frontend

Open a new terminal:

```bash
cd frontend
pip install -r requirements.txt

# Create secrets file
mkdir -p .streamlit
echo 'API_URL = "http://localhost:8000"' > .streamlit/secrets.toml

# Start Streamlit
streamlit run app.py
```

Frontend will be available at: `http://localhost:8501`

## Step 5: Use the Application

1. **Prediction Page**:
   - Upload your processed CSV file
   - Select a row
   - Click "Predict" to get price prediction

2. **Backtesting Page**:
   - Adjust training set proportion
   - Click "Run Backtest"
   - View performance metrics and charts

3. **Explainability Page**:
   - Upload data or enter features manually
   - Click "Explain Prediction"
   - View SHAP values and feature importance

## Troubleshooting

### Models Not Loading

If you see "models_not_loaded" in health check:

1. Check that models were trained:
   ```bash
   ls backend/models/
   ```

2. Check model paths in `backend/.env` or environment variables

3. Verify model files exist:
   ```bash
   find backend/models -name "*.json" -o -name "*.pth"
   ```

### Frontend Can't Connect

1. Verify backend is running:
   ```bash
   curl http://localhost:8000/api/v1/health
   ```

2. Check `frontend/.streamlit/secrets.toml`:
   ```toml
   API_URL = "http://localhost:8000"
   ```

3. Check CORS settings in `backend/app/main.py`

### Import Errors

If you see import errors:

```bash
# Reinstall dependencies
cd backend
pip install -r requirements.txt --force-reinstall

cd ../frontend
pip install -r requirements.txt --force-reinstall
```

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design details
- Read [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment
- Check [README.md](README.md) for full documentation

## Common Commands

```bash
# Train models
cd backend && python train_model.py

# Run backend
cd backend && uvicorn app.main:app --reload

# Run frontend
cd frontend && streamlit run app.py

# Check API health
curl http://localhost:8000/api/v1/health

# View API docs
open http://localhost:8000/docs
```

## File Structure After Setup

```
Stock_Prediction_Model/
├── backend/
│   ├── app/              # FastAPI application
│   ├── models/           # Trained models (after training)
│   ├── data/             # Processed data files
│   └── train_model.py    # Training script
├── frontend/
│   ├── app.py            # Streamlit app
│   └── utils/            # Frontend utilities
├── Stock_prediction.ipynb  # Original notebook (preserved)
└── [data files]          # Original data files
```

## Support

For issues or questions:
1. Check the logs in terminal
2. Review error messages in the UI
3. Check API documentation at `/docs` endpoint

