"""
FastAPI application entry point.
Production-grade stock prediction API.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn

from app.config import settings
from app.api.routes import router
from app.services.model_registry import ModelRegistry
from app.services.feature_engineering import FeatureEngineer

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Production-grade stock prediction API with ensemble models"
)

# CORS middleware (configure for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting up Stock Prediction API...")
    
    try:
        # Initialize model registry and load models
        registry = ModelRegistry()
        
        # Try to load models
        try:
            registry.load_models()
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load models on startup: {e}")
            logger.warning("Models will be loaded on first request")
        
        # Store registry in app state for dependency injection
        app.state.model_registry = registry
        
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Stock Prediction API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )

