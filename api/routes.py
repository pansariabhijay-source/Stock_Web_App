"""
api/routes.py — All API endpoint handlers.

ENDPOINTS:
  POST /api/predict   → run model prediction for a stock + horizon
  POST /api/explain   → get SHAP feature importances
  POST /api/backtest  → run trading simulation
  GET  /api/models    → list all available stocks + models
  GET  /api/regime    → current market regime
  GET  /api/health    → health check

HOW FASTAPI ROUTING WORKS:
  Each function decorated with @router.get() or @router.post() becomes
  an API endpoint. FastAPI automatically:
  - Parses the request body into the Pydantic schema
  - Validates all fields
  - Returns 422 if validation fails
  - Serializes the return value to JSON
  - Documents everything at /docs
"""

import logging
from datetime import datetime
from typing import Dict

import numpy as np
from fastapi import APIRouter, HTTPException

from api.schemas import (
    PredictRequest, PredictResponse, PredictionDetail,
    ExplainRequest, ExplainResponse, FeatureImportance,
    BacktestRequest, BacktestResponse, BacktestMetrics,
    ModelsResponse, ModelInfo, HealthResponse,
)
from api.model_registry import ModelRegistry
from config import LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

router = APIRouter()

# Global registry — loaded once at startup in main.py
registry: ModelRegistry = None


def set_registry(reg: ModelRegistry):
    """Called from main.py to inject the loaded registry."""
    global registry
    registry = reg


def _check_registry():
    if registry is None:
        raise HTTPException(status_code=503, detail="Model registry not loaded yet")


# ── Predict ────────────────────────────────────────────────────────────────────

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Get a stock price direction prediction.

    Returns the model's prediction (UP/DOWN), probability, expected return,
    and confidence interval for the requested horizon.
    """
    _check_registry()

    ticker   = request.ticker.upper()
    horizon  = request.horizon
    model_id = request.model

    # Validate ticker
    if ticker not in registry.available_tickers:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for {ticker}. "
                   f"Available: {sorted(registry.available_tickers)[:5]}..."
        )

    # Validate horizon
    if horizon not in ["1d", "5d", "20d"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid horizon '{horizon}'. Must be one of: 1d, 5d, 20d"
        )

    try:
        result = registry.predict(ticker, horizon, model_id)
    except Exception as e:
        logger.error(f"Prediction failed for {ticker}/{horizon}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Determine signal strength from probability
    prob = result["probability"]
    dist_from_center = abs(prob - 0.5)
    if dist_from_center >= 0.15:
        signal_strength = "strong"
    elif dist_from_center >= 0.08:
        signal_strength = "moderate"
    else:
        signal_strength = "weak"

    # Get meta info
    meta = registry.get_stock_meta(ticker)

    return PredictResponse(
        ticker       = ticker,
        company_name = meta.get("name", ticker),
        sector       = meta.get("sector", "Unknown"),
        horizon      = horizon,
        current_price= result.get("current_price", 0.0),
        prediction   = PredictionDetail(
            direction        = "UP" if prob > 0.5 else "DOWN",
            probability      = round(prob, 4),
            predicted_return = round(result.get("predicted_return", 0.0), 4),
            confidence_lower = round(result.get("confidence_lower", 0.0), 4),
            confidence_upper = round(result.get("confidence_upper", 0.0), 4),
            signal_strength  = signal_strength,
        ),
        regime       = result.get("regime", "unknown"),
        model_used   = model_id,
        last_updated = datetime.now().isoformat(),
    )


# ── Explain ────────────────────────────────────────────────────────────────────

@router.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    """
    Get SHAP-based feature importance for the latest prediction.

    Returns the top N features that drove the model's prediction,
    with their contribution direction (positive = pushed toward UP).
    """
    _check_registry()

    ticker  = request.ticker.upper()
    horizon = request.horizon
    top_n   = request.top_n

    if ticker not in registry.available_tickers:
        raise HTTPException(status_code=404, detail=f"No model for {ticker}")

    try:
        shap_result = registry.explain(ticker, horizon, top_n)
    except Exception as e:
        logger.error(f"Explain failed for {ticker}/{horizon}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    features = [
        FeatureImportance(
            feature    = f["feature"],
            importance = round(f["importance"], 6),
            direction  = f["direction"],
        )
        for f in shap_result["top_features"]
    ]

    # Build plain-English interpretation
    top3 = [f.feature for f in features[:3]]
    direction = "upward" if shap_result.get("net_direction", 0) > 0 else "downward"
    interpretation = (
        f"The model is biased {direction} primarily due to: "
        f"{', '.join(top3)}. "
        f"These features had the largest influence on this prediction."
    )

    return ExplainResponse(
        ticker         = ticker,
        horizon        = horizon,
        top_features   = features,
        interpretation = interpretation,
    )


# ── Backtest ───────────────────────────────────────────────────────────────────

@router.post("/backtest", response_model=BacktestResponse)
async def backtest(request: BacktestRequest):
    """
    Run a simulated trading strategy using model predictions.

    Simulates buying when model predicts UP and going to cash when DOWN.
    Returns performance metrics: Sharpe ratio, max drawdown, annual return etc.
    """
    _check_registry()

    ticker   = request.ticker.upper()
    horizon  = request.horizon
    tc       = request.transaction_cost

    if ticker not in registry.available_tickers:
        raise HTTPException(status_code=404, detail=f"No model for {ticker}")

    try:
        bt_result = registry.backtest(ticker, horizon, tc)
    except Exception as e:
        logger.error(f"Backtest failed for {ticker}/{horizon}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    metrics = bt_result["metrics"]

    # Plain-English summary
    sharpe    = metrics["sharpe_ratio"]
    ann_ret   = metrics["annual_return"]
    bench_ret = metrics["benchmark_annual_return"]
    hit_rate  = metrics["hit_rate"]

    if sharpe > 1.5:
        quality = "strong"
    elif sharpe > 0.8:
        quality = "decent"
    else:
        quality = "weak"

    outperform = "outperforms" if ann_ret > bench_ret else "underperforms"
    summary = (
        f"The {quality} strategy achieves {ann_ret:.1%} annual return "
        f"with a Sharpe ratio of {sharpe:.2f}. "
        f"It {outperform} the buy-and-hold benchmark ({bench_ret:.1%}) "
        f"with a {hit_rate:.1%} directional accuracy."
    )

    return BacktestResponse(
        ticker  = ticker,
        horizon = horizon,
        metrics = BacktestMetrics(**metrics),
        summary = summary,
    )


# ── Models List ────────────────────────────────────────────────────────────────

@router.get("/models", response_model=ModelsResponse)
async def list_models():
    """
    List all stocks with trained models.

    Returns company names, sectors, available horizons,
    and best accuracy per horizon for each stock.
    """
    _check_registry()

    stocks = []
    for ticker in sorted(registry.available_tickers):
        meta     = registry.get_stock_meta(ticker)
        horizons = registry.get_available_horizons(ticker)
        accuracy = registry.get_best_accuracy(ticker)

        stocks.append(ModelInfo(
            ticker              = ticker,
            company_name        = meta.get("name", ticker),
            sector              = meta.get("sector", "Unknown"),
            horizons_available  = horizons,
            best_accuracy       = accuracy,
        ))

    return ModelsResponse(
        total_stocks = len(stocks),
        stocks       = stocks,
    )


# ── Regime ─────────────────────────────────────────────────────────────────────

@router.get("/regime")
async def get_regime():
    """
    Get the current market regime.

    Returns the HMM-detected market regime: bull, bear, sideways, or crisis.
    Updated on each prediction call using the latest Nifty index data.
    """
    _check_registry()

    try:
        regime_info = registry.get_current_regime()
        return {
            "regime":      regime_info["regime"],
            "description": _regime_description(regime_info["regime"]),
            "since":       regime_info.get("since", "unknown"),
            "duration_days": regime_info.get("duration_days", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _regime_description(regime: str) -> str:
    descriptions = {
        "bull":     "Markets trending upward with low volatility. Momentum strategies work well.",
        "bear":     "Markets trending downward with elevated fear. Capital preservation is key.",
        "sideways": "No clear trend. Range-bound choppy action. Breakout strategies preferred.",
        "crisis":   "Extreme volatility and panic. All correlations spike. High uncertainty.",
    }
    return descriptions.get(regime, "Unknown market state.")


# ── Health ─────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check — confirms API is running and models are loaded."""
    if registry is None:
        return HealthResponse(
            status           = "degraded",
            models_loaded    = 0,
            stocks_available = 0,
        )

    return HealthResponse(
        status           = "healthy",
        models_loaded    = registry.total_models_loaded,
        stocks_available = len(registry.available_tickers),
    )
