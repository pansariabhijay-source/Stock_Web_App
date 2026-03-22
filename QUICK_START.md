# Quick Start Guide - Production Models

## Step 1: Install LightGBM

```bash
cd backend
pip install lightgbm>=4.1.0
```

## Step 2: Train New Models

```bash
python train_models_production.py
```

This will:
- Train LightGBM, XGBoost, and LSTM models
- Evaluate on test set
- Save models and register them
- Show performance metrics

**Expected time**: 5-15 minutes depending on your hardware

## Step 3: Restart Backend

```bash
# Stop current backend (Ctrl+C)
# Then restart
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Step 4: Test Backtesting

1. Open frontend
2. Go to Backtesting page
3. Click "Run Backtest"
4. **Expected RMSE**: 50-500 (not millions!)

## What Changed?

### Before:
- ❌ RMSE: ~650,000,000 (wrong target comparison)
- ❌ Single model (XGBoost + NN residual)
- ❌ Basic metrics

### After:
- ✅ RMSE: ~50-500 (correct target comparison)
- ✅ Multiple SOTA models (LightGBM + XGBoost + LSTM)
- ✅ Comprehensive metrics (Sharpe, Sortino, MASE, etc.)
- ✅ Better predictions with ensemble

## Troubleshooting

### If training fails:
1. Check data file exists: `backend/data/RILO - Copy.csv`
2. Verify all dependencies installed: `pip install -r requirements.txt`
3. Check Python version: 3.8+

### If backtesting still shows high RMSE:
1. Verify new models trained successfully
2. Check backend logs for errors
3. Ensure Target_Close column exists in data
4. Check model registry has correct active version

## Model Comparison

After training, you'll see which model performs best:
- **LightGBM**: Usually fastest and most accurate
- **XGBoost**: Robust baseline
- **LSTM**: Captures temporal patterns
- **Ensemble**: Best overall (weighted combination)

---

**Ready to go!** Train the models and enjoy production-grade predictions! 🚀

