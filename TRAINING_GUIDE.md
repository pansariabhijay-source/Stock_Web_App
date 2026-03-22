# Production-Grade Model Training Guide

## New Training Script

Use `train_models_production.py` instead of `train_model.py` for production-grade models.

## Models Included

1. **LightGBM** - Fast, accurate gradient boosting (preferred)
2. **XGBoost** - Robust baseline with optimized hyperparameters
3. **LSTM** - Deep learning for time-series patterns with attention mechanism

## Training Command

```bash
cd backend
python train_models_production.py
```

## What's Improved

### 1. Better Hyperparameters
- Optimized learning rates (0.01 instead of 0.003)
- Better regularization
- Early stopping with patience
- Feature and bagging fractions tuned

### 2. Proper Validation
- Train/Val/Test split (time-series aware)
- No data leakage
- Proper evaluation on held-out test set

### 3. Multiple Models
- LightGBM (fastest, often most accurate)
- XGBoost (robust baseline)
- LSTM (captures temporal patterns)

### 4. Ensemble
- Weighted ensemble: 40% LightGBM + 40% XGBoost + 20% LSTM
- Automatically uses best available models

## Expected Results

After training, you should see:
- **LightGBM RMSE**: ~50-200 (depending on data)
- **XGBoost RMSE**: ~50-250
- **LSTM RMSE**: ~60-300
- **Ensemble RMSE**: Best of all (often 10-20% better than individual models)

## Backtesting Fix

The backtesting now:
- ✅ Uses Target_Close (next-day prediction) instead of Close Price
- ✅ Validates predictions are reasonable
- ✅ Uses exact model feature names
- ✅ Better error handling and logging

## Next Steps

1. **Train new models**: Run `python train_models_production.py`
2. **Test backtesting**: The RMSE should now be reasonable
3. **Compare models**: Check which performs best on your data
4. **Fine-tune**: Adjust hyperparameters if needed

## Troubleshooting

If RMSE is still high:
1. Check that models are using correct features
2. Verify Target_Close column exists and has reasonable values
3. Check model predictions are in same range as actuals
4. Review training logs for any warnings

