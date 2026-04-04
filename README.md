<h1 align="center">AlphaStock Terminal</h1>
<p align="center">
  <strong>AI-Powered Institutional Intelligence for Nifty 50 Equities</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/FastAPI-0.109+-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/TailwindCSS-3.4-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</p>

---

AlphaStock Terminal is a full-stack, AI-driven financial analytics platform that delivers institutional-grade predictive intelligence for **NIFTY 50** Indian market equities. It combines ensemble machine learning models (LightGBM, XGBoost), Hidden Markov Model-based market regime detection, and SHAP explainability — all served through a high-performance FastAPI backend and a premium dark-mode React dashboard.

---

## ✨ Features

### 🧠 Machine Learning Predictions
- **Multi-Horizon Forecasting** — Predict stock direction (UP/DOWN) across **1-day**, **5-day**, and **20-day** horizons with confidence scores.
- **Ensemble Models** — Parallel LightGBM and XGBoost classifier/regressor ensembles trained on 10+ years of historical NSE data.
- **Signal Strength** — Each prediction is classified as `strong`, `moderate`, or `weak` based on probability distance from center.

### 🔎 Explainable AI (SHAP)
- **Feature Attribution** — SHAP (SHapley Additive exPlanations) provides transparent, interpretable feature-level explanations for every prediction.
- **Plain-English Interpretation** — Automatically generated human-readable summaries explain *why* the model predicted what it did.

### 📊 Market Regime Detection
- **HMM-Based Regime Classification** — Detects whether the market is in a `Bull`, `Bear`, `Sideways`, or `Crisis` state using Hidden Markov Models applied to the Nifty index.

### 📈 Backtesting Engine
- **Strategy Simulation** — Backtest model predictions against historical data with configurable transaction costs.
- **Rich Metrics** — Sharpe ratio, max drawdown, annual return, hit rate, and benchmark comparison.

### 🖥️ Premium Dashboard
- **Institutional Dark Theme** — Sleek bento-grid layouts with glassmorphism effects and Framer Motion micro-animations.
- **Live Ticker Tape** — Seamless animated marquee tracking real-time prices and percentage changes across all loaded stocks.
- **Interactive Trend Charts** — Recharts-powered charts with historical data overlay and predicted trajectory projections.
- **Top Gainers & Losers** — Dynamic ranking panels driven by actual model predictions.

<!-- 🔗 Live Demo: [Add link here] -->

---

## 🏗️ Architecture

```
alpha_stock/
├── api/                        # FastAPI application
│   ├── main.py                 # App entry point, lifespan, CORS, routes
│   ├── routes.py               # All API endpoint handlers
│   ├── model_registry.py       # Model loading, prediction, SHAP, backtesting
│   └── schemas.py              # Pydantic request/response models
│
├── models/                     # ML model definitions
│   ├── base.py                 # Base model interface
│   ├── classifier.py           # Classification model wrappers
│   ├── lgbm_xgb.py             # LightGBM & XGBoost implementations
│   └── ensemble.py             # Ensemble model combining multiple learners
│
├── features/                   # Feature engineering pipeline
│   ├── pipeline.py             # Full feature pipeline orchestration
│   ├── technical.py            # 130+ technical indicators (RSI, MACD, BB, etc.)
│   ├── regime.py               # HMM market regime detection
│   └── selection.py            # Feature importance & selection
│
├── data_pipeline/              # Data ingestion & management
│   ├── ingestion.py            # yFinance OHLCV data downloader
│   ├── nifty50.py              # Nifty 50 universe management
│   └── news_sentiment.py       # FinBERT-based news sentiment analysis
│
├── training/                   # Model training & evaluation
│   ├── trainer.py              # Core training loop with walk-forward CV
│   ├── train_all.py            # Batch training script for all tickers
│   └── backtest.py             # Historical backtesting engine
│
├── frontend/                   # React + Vite frontend
│   └── src/
│       ├── App.jsx             # Root component with routing
│       ├── api.js              # Axios API client
│       ├── pages/
│       │   ├── Dashboard.jsx   # Main dashboard with forecasts & charts
│       │   ├── Screener.jsx    # Nifty 50 stock screener table
│       │   ├── Analysis.jsx    # Deep-dive stock analysis page
│       │   ├── Engine.jsx      # Technical whitepaper / architecture page
│       │   ├── Portfolio.jsx   # Portfolio tracker (WIP)
│       │   └── About.jsx       # Project information & credits
│       └── components/
│           ├── LiveTickerTape.jsx   # Animated scrolling ticker
│           ├── HeroForecast.jsx     # Bull/Bear regime hero card
│           ├── ForecastBentoGrid.jsx # Top prediction cards grid
│           ├── TrendChart.jsx       # Interactive price/forecast chart
│           ├── MoversList.jsx       # Top gainers/losers panels
│           ├── SHAPModal.jsx        # SHAP feature importance modal
│           ├── Header.jsx           # Page header
│           ├── Sidebar.jsx          # Navigation sidebar
│           └── LiveTicker.jsx       # Individual ticker component
│
├── config.py                   # Central configuration (paths, tickers, hyperparams)
├── requirements.txt            # Python dependencies
└── .env                        # API keys (NEWS_API_KEY, HF_TOKEN)
```

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend** | React 19, Vite 8, TailwindCSS 3.4, Framer Motion, Recharts, Lucide React, Axios, React Router 7 |
| **Backend** | Python 3.9+, FastAPI, Uvicorn, Pydantic V2 |
| **Machine Learning** | Scikit-Learn, LightGBM, XGBoost, SHAP, Optuna (hyperparameter tuning) |
| **Regime Detection** | hmmlearn (Hidden Markov Models) |
| **NLP / Sentiment** | Hugging Face Transformers (FinBERT) |
| **Data** | yFinance, Pandas, NumPy, PyArrow, pandas-ta (130+ technical indicators) |
| **Deep Learning** | PyTorch, PyTorch Lightning, PyTorch Forecasting (TFT) |

---

## 🔌 API Endpoints

All routes are prefixed with `/api`. Interactive Swagger docs available at [`/docs`](http://localhost:8000/docs).

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/predict` | Get directional prediction (UP/DOWN) with probability, expected return, and confidence interval |
| `POST` | `/api/explain` | Get SHAP-based top-N feature importances with plain-English interpretation |
| `POST` | `/api/backtest` | Run simulated trading strategy and get Sharpe ratio, max drawdown, hit rate |
| `GET` | `/api/models` | List all available stocks with trained models, sectors, and accuracy |
| `GET` | `/api/prices` | Get current prices and 1-day % change for all loaded stocks |
| `GET` | `/api/history/{ticker}` | Get historical closing prices for charting (configurable `days` param) |
| `GET` | `/api/regime` | Get HMM-detected market regime (bull/bear/sideways/crisis) |
| `GET` | `/api/health` | Health check — confirms API is running and models are loaded |

---

## 🚀 Getting Started

### Prerequisites

- **Python** 3.9+
- **Node.js** v18+
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/pansariabhijay-source/Stock_Web_App.git
cd Stock_Web_App
```

### 2. Backend Setup (FastAPI)

```bash
# Create and activate virtual environment
python -m venv env

# Windows
.\env\Scripts\activate

# macOS/Linux
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys
echo NEWS_API_KEY=your_newsapi_key > .env
echo HF_TOKEN=your_huggingface_token >> .env

# Launch FastAPI server
python -m uvicorn api.main:app --reload --port 8000
```

> 📖 **API Docs**: Once running, visit [`http://localhost:8000/docs`](http://localhost:8000/docs) for interactive Swagger documentation.

### 3. Frontend Setup (Vite + React)

Open a **new terminal** session:

```bash
cd frontend
npm install
npm run dev
```

> 🌐 **Dashboard**: Open [`http://localhost:5173`](http://localhost:5173) in your browser.

---

## 📡 Configuration

All settings are centralized in [`config.py`](config.py):

| Setting | Description | Default |
|---|---|---|
| `NIFTY50_TICKERS` | Full universe of 50 NSE-listed stocks | All Nifty 50 constituents |
| `HISTORICAL_START` | Training data start date | `2015-01-01` |
| `HORIZONS` | Prediction windows | `1d`, `5d`, `20d` |
| `TEST_RATIO` | Hold-out test set proportion | `15%` |
| `N_CV_SPLITS` | Walk-forward CV folds | `5` |
| `N_OPTUNA_TRIALS` | Hyperparameter tuning budget | `50` |
| `CACHE_TTL_SECONDS` | Redis cache TTL | `3600s` |

---

## 🗺️ Roadmap

- [ ] Migrate `yfinance` to institutional-grade WebSocket provider (Polygon.io / TrueData)
- [ ] Implement Time-Series Database (TimescaleDB / InfluxDB) for persistent storage
- [ ] Separate data processing pipelines using Celery/Redis message queuing
- [ ] Decouple model serving using NVIDIA Triton or MLflow
- [ ] Enforce JWT Authentication and tighten CORS constraints
- [ ] Add real-time WebSocket price streaming to the frontend
- [ ] Deploy to Render with CI/CD pipeline

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Built with ☕ by <strong>Abhijay Pansari</strong><br/>
  <a href="https://github.com/pansariabhijay-source/Stock_Web_App">⭐ Star this repo if you find it useful!</a>
</p>
