# AlphaStock Terminal | Institutional Intelligence


<p align="left">
  <img src="https://img.shields.io/badge/Python-3.9+-blue" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-green" />
  <img src="https://img.shields.io/badge/Frontend-React%2019-black" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange" />
  <img src="https://img.shields.io/badge/Models-RF%20%7C%20LR%20%7C%20DT%20%7C%20SVM-purple" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

---

## Description

AlphaStock Terminal is a high-performance financial analytics platform designed to replicate institutional-grade trading dashboards. It integrates real-time market tracking with machine learning-driven predictions and explainability systems.

The system combines a modern reactive frontend with a scalable FastAPI backend to deliver probabilistic forecasts and actionable insights on NIFTY 50 equities.

---

## Features

### Market Intelligence
- Real-time equity tracking with dynamic price updates  
- Infinite ticker tape for continuous monitoring  
- Volatility-aware visual feedback  

### Predictive Modeling
- Ensemble models using LightGBM and XGBoost  
- Multi-horizon predictions (1-day, 5-day, 20-day)  
- Expected return estimation with probability scoring  

### Explainable AI
- SHAP-based interpretability  
- Feature importance visualization  
- Transparent reasoning behind predictions  

### UI and Experience
- Institutional dark-mode dashboard  
- Bento-grid layout inspired by trading terminals  
- Smooth animations using Framer Motion  

---

## System Architecture

### Frontend
- React 19 (Vite)
- TailwindCSS
- Recharts

### Backend
- FastAPI
- Uvicorn
- RESTful APIs
- Asynchronous processing

### Machine Learning Layer
- Scikit-learn pipelines
- LightGBM and XGBoost
- SHAP for explainability

### Data Layer
- yFinance for OHLCV and fundamentals

## Project Structure
AlphaStock-Terminal/
│
├── api/ # FastAPI backend
├── frontend/ # React application
├── models/ # ML models
├── dashboard/ # UI assets
├── requirements.txt
└── README.md

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- Git

---

## Getting Started

### Backend Setup

```bash
python -m venv env
env\Scripts\activate

pip install -r requirements.txt

python -m uvicorn api.main:app --reload --port 8000

## Frontend Setup

```bash
cd frontend
npm install
npm run dev

npm install

npm run dev
