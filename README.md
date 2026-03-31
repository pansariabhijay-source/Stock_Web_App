# AlphaStock Terminal | Institutional Intelligence

AlphaStock Terminal is a sophisticated, Amazon-tier interactive financial dashboard designed for market analysis, predictive modeling, and real-time equity tracking using high-performance machine learning.

![AlphaStock Terminal UI](./dashboard/preview.png)

## Overview

This project seamlessly combines a modern reactive frontend with a resilient FastAPI backend capable of making real-time stochastic and ML-driven predictions for NIFTY 50 Indian market equities.

### Key Capabilities

- **Institutional Design Philosophy**: Modern dark-mode bento-box grid layouts with interactive Framer Motion elements, mimicking high-value proprietary terminal software.
- **Micro-Predictions**: LightGBM and XGBoost classifier/regressor ensembles provide dynamic probability scores and expected percentage returns for 1-day, 5-day, and 20-day horizons.
- **Explainable AI (XAI)**: SHAP (SHapley Additive exPlanations) values provide immediate interpretability for *why* the model made specific predictions (i.e., top impacting features).
- **Infinite Ticket Tape**: Seamless, animated `<LiveTickerTape />` dynamically tracking price ticks and highlighting moment-to-moment volatility logic.

## Stack Arsenal

| Area | Technologies Used |
|---|---|
| Frontend | React 19, Vite, TailwindCSS v4, Framer Motion, Recharts, Lucide React, Axios |
| Backend | Python 3, FastAPI, Uvicorn |
| Machine Learning | Scikit-Learn, LightGBM, XGBoost, SHAP, Pandas, PyArrow |
| Data Integration | yFinance for historical OHLCV/Fundamentals |

---

## Developer Setup

Currently, this repository represents the fully-working local developer environment MVP.

### 1. Prerequisites 

- Python 3.9+
- Node.js v18+ 
- Git

### 2. Backend Orchestration (FastAPI)

\`\`\`bash
# Create and activate virtual environment (Windows)
python -m venv env
.\\env\\Scripts\\activate

# Install requirements
pip install -r requirements.txt

# Launch FastAPI on localhost:8000
python -m uvicorn api.main:app --reload --port 8000
\`\`\`
*Access the Swagger OpenAPI Documentation at [`http://localhost:8000/docs`](http://localhost:8000/docs)*

### 3. Frontend Terminal (Vite/React)

Open a new terminal session.

\`\`\`bash
cd frontend

# Install Node modules
npm install

# Instigate Vite development server at localhost:5173
npm run dev
\`\`\`

---

## Roadmap

As part of transitioning from 'Development Grade' to 'Production Grade', the following upgrades are slated:
- [ ] Migrate `yfinance` to institutional-grade Websocket provider (e.g., Polygon.io, TrueData). 
- [ ] Implement backend Time-Series Database (TimescaleDB / InfluxDB).
- [ ] Separate Data Processing pipelines using Celery/Redis message queuing. 
- [ ] Decouple Model Serving using NVIDIA Triton or MLflow.
- [ ] Enforce rigid JWT Authentication and tighten CORS constraints. 

---

*Made by Abhijay Pansari*
