# Stock Prediction Frontend

## Overview

Streamlit-based frontend for the Stock Prediction API. Provides:
- Interactive candlestick charts with predictions
- SHAP feature importance visualization
- Backtesting results and equity curves
- Model comparison and confidence indicators
- Real-time prediction interface

## Architecture

```
frontend/
├── app.py                 # Main Streamlit application
├── pages/
│   ├── prediction.py     # Prediction interface
│   ├── backtesting.py    # Backtesting visualization
│   └── explainability.py # SHAP explanations
├── utils/
│   ├── api_client.py     # FastAPI client
│   ├── charts.py         # Chart utilities
│   └── data_loader.py    # Data loading helpers
├── requirements.txt
└── .streamlit/
    └── config.toml       # Streamlit configuration
```

## Features

1. **Prediction Dashboard**
   - Input feature vector or load from data
   - Display prediction with confidence intervals
   - Overlay predictions on historical price chart

2. **Backtesting Visualization**
   - Equity curve
   - Drawdown analysis
   - Performance metrics (RMSE, direction accuracy)
   - Prediction vs actual comparison

3. **Explainability**
   - SHAP waterfall plots
   - Feature importance rankings
   - Prediction-level explanations

4. **Model Management**
   - View available model versions
   - Compare model performance
   - Switch between models

## Usage

```bash
streamlit run app.py
```

Configure API endpoint in `.streamlit/config.toml` or environment variables.

