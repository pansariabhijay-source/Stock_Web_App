"""
api/main.py — FastAPI application entry point.

HOW TO RUN:
  Development (auto-reload on code changes):
    uvicorn api.main:app --reload --port 8000

  Production (Render will use this):
    uvicorn api.main:app --host 0.0.0.0 --port 8000

WHAT HAPPENS AT STARTUP:
  1. FastAPI app is created
  2. CORS is configured (allows React frontend to call the API)
  3. Model registry loads all trained models into memory
  4. API is ready to serve requests

INTERACTIVE DOCS:
  Once running, visit:
    http://localhost:8000/docs      → Swagger UI (try endpoints live)
    http://localhost:8000/redoc    → ReDoc (clean documentation)
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router, set_registry
from api.model_registry import ModelRegistry
from config import LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager — runs startup code before the app serves requests.

    WHY LIFESPAN (not @app.on_event):
      The modern FastAPI way. Cleaner than deprecated event handlers.
      Everything before `yield` runs at startup.
      Everything after `yield` runs at shutdown.
    """
    # ── Startup ────────────────────────────────────────────────────────────────
    logger.info("AlphaStock API starting up...")

    registry = ModelRegistry()
    registry.load_all()
    set_registry(registry)

    logger.info(f"API ready — {len(registry.available_tickers)} stocks loaded")
    logger.info("Docs available at: http://localhost:8000/docs")

    yield   # API serves requests here

    # ── Shutdown ───────────────────────────────────────────────────────────────
    logger.info("AlphaStock API shutting down...")


# Create FastAPI app
app = FastAPI(
    title       = "AlphaStock Prediction API",
    description = """
    AI-powered stock price prediction for Nifty 50 stocks.

    Features:
    - Direction predictions (UP/DOWN) with confidence scores
    - 3 prediction horizons: 1 day, 5 days, 20 days
    - SHAP-based explainability: know WHY the model predicted what it did
    - Market regime detection: bull / bear / sideways / crisis
    - Backtesting: see how the model would have performed historically

    Built with LightGBM, XGBoost, and ensemble methods trained on 10 years of NSE data.
    """,
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── CORS Middleware ────────────────────────────────────────────────────────────
# CORS = Cross-Origin Resource Sharing
# Without this, the browser blocks React (port 5173) from calling FastAPI (port 8000)
# because they're on different ports = different "origins"

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",    # Vite dev server
        "http://localhost:3000",    # Create React App (if used)
        "http://127.0.0.1:5173",
        "https://*.onrender.com",   # Render deployment
        "*",                        # Allow all for now — restrict in production
    ],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Register Routes ────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api")

# ── Root Endpoint ──────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name":    "AlphaStock Prediction API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/api/health",
        "status":  "running",
    }
