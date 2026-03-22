# Deployment Guide

## Backend Deployment (Render/Railway)

### Prerequisites
- GitHub repository with your code
- Render/Railway account
- Trained models in `backend/models/`

### Steps

#### 1. Prepare Backend

```bash
cd backend
# Ensure requirements.txt is up to date
# Ensure models are trained and saved
```

#### 2. Deploy to Render

1. **Create New Web Service**:
   - Connect GitHub repository
   - Select `backend` directory
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

2. **Environment Variables**:
   ```
   API_TITLE=Stock Prediction API
   DEBUG=False
   LOG_LEVEL=INFO
   XGBOOST_MODEL_PATH=models/v1/xgboost_model.json
   NN_MODEL_PATH=models/v1/neural_network
   ```

3. **Add Build Script** (optional):
   Create `render_build.sh`:
   ```bash
   #!/bin/bash
   pip install -r requirements.txt
   # Download models if stored remotely
   ```

#### 3. Deploy to Railway

1. **Create New Project**:
   - Connect GitHub repository
   - Select `backend` directory

2. **Configure Service**:
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Add environment variables (same as Render)

3. **Deploy Models**:
   - Option 1: Include in repository (not recommended for large models)
   - Option 2: Use Railway volumes
   - Option 3: Store in S3/GCS and download on startup

### Model Storage Options

#### Option 1: Include in Repository
```bash
# Models are committed to git
git add backend/models/
git commit -m "Add trained models"
```

#### Option 2: External Storage (Recommended)
```python
# In app/main.py startup, download models from S3
import boto3

s3 = boto3.client('s3')
s3.download_file('your-bucket', 'models/xgboost.json', 'models/xgboost.json')
```

#### Option 3: Model Registry Service
- Use MLflow
- Use custom model registry API
- Download on first request

## Frontend Deployment (Streamlit Cloud)

### Steps

1. **Prepare Frontend**:
   ```bash
   cd frontend
   # Ensure requirements.txt is up to date
   ```

2. **Deploy to Streamlit Cloud**:
   - Go to share.streamlit.io
   - Connect GitHub repository
   - Select `frontend` directory
   - Main file: `app.py`

3. **Configure Secrets**:
   - In Streamlit Cloud dashboard → Settings → Secrets
   - Add:
     ```toml
     API_URL=https://your-backend-url.onrender.com
     ```

4. **Deploy**:
   - Click "Deploy"
   - Wait for build to complete
   - Access your app at `https://your-app.streamlit.app`

## Local Development Setup

### Backend

```bash
cd backend
pip install -r requirements.txt

# Train models (if not already done)
python train_model.py

# Run server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
pip install -r requirements.txt

# Create secrets file
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml and set API_URL=http://localhost:8000

# Run Streamlit
streamlit run app.py
```

## Docker Deployment (Alternative)

### Backend Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - API_TITLE=Stock Prediction API
      - DEBUG=False
    volumes:
      - ./backend/models:/app/models
      - ./backend/data:/app/data

  frontend:
    build: ./frontend
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://backend:8000
    depends_on:
      - backend
```

## Environment Variables Reference

### Backend

| Variable | Description | Default |
|----------|-------------|---------|
| `API_TITLE` | API title | "Stock Prediction API" |
| `DEBUG` | Debug mode | False |
| `XGBOOST_MODEL_PATH` | Path to XGBoost model | None |
| `NN_MODEL_PATH` | Path to NN model | None |
| `REDIS_HOST` | Redis host | localhost |
| `REDIS_PORT` | Redis port | 6379 |
| `LOG_LEVEL` | Logging level | INFO |

### Frontend

| Variable | Description | Default |
|----------|-------------|---------|
| `API_URL` | Backend API URL | http://localhost:8000 |

## Troubleshooting

### Backend Issues

1. **Models not loading**:
   - Check model paths in environment variables
   - Verify models exist in specified paths
   - Check logs for loading errors

2. **Import errors**:
   - Ensure all dependencies in requirements.txt
   - Check Python version (3.9+)

3. **Port conflicts**:
   - Change port in uvicorn command
   - Update frontend API_URL

### Frontend Issues

1. **Cannot connect to API**:
   - Check API_URL in secrets.toml
   - Verify backend is running
   - Check CORS settings

2. **Charts not displaying**:
   - Check Plotly installation
   - Verify data format

## Production Checklist

- [ ] Models trained and saved
- [ ] Environment variables configured
- [ ] Secrets properly managed
- [ ] CORS configured for production
- [ ] Logging configured
- [ ] Monitoring set up
- [ ] Error handling tested
- [ ] API documentation accessible
- [ ] Health check endpoint working
- [ ] Frontend connected to backend

