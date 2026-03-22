# Installation Guide

This guide will help you install all required dependencies for both the backend and frontend.

## Prerequisites

- **Python 3.9 or higher** (Python 3.11 recommended)
- **pip** (Python package manager)
- **Virtual environment** (recommended)

## Step 1: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

## Step 2: Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Backend Dependencies Explained

- **fastapi**: Web framework for building APIs
- **uvicorn**: ASGI server for running FastAPI
- **pydantic**: Data validation using Python type annotations
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning utilities (scalers, metrics)
- **xgboost**: Gradient boosting framework
- **torch**: PyTorch for neural networks
- **shap**: SHAP values for model explainability
- **redis**: Caching (optional, can be commented out)
- **python-dotenv**: Environment variable management
- **python-multipart**: File upload support

### Optional Backend Dependencies

If you don't need Redis caching or Prometheus monitoring, you can comment out these lines in `requirements.txt`:
- `redis==5.0.1`
- `prometheus-client==0.19.0`

## Step 3: Install Frontend Dependencies

```bash
cd frontend
pip install -r requirements.txt
```

### Frontend Dependencies Explained

- **streamlit**: Web framework for building the dashboard
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **plotly**: Interactive charts and visualizations
- **requests**: HTTP library for API calls

## Step 4: Verify Installation

### Backend Verification

```bash
cd backend
python -c "import fastapi, uvicorn, xgboost, torch, shap; print('All backend dependencies installed!')"
```

### Frontend Verification

```bash
cd frontend
python -c "import streamlit, plotly, requests; print('All frontend dependencies installed!')"
```

## Troubleshooting

### Common Issues

#### 1. **PyTorch Installation Fails**

PyTorch installation can be platform-specific. If the default installation fails:

**For CPU-only:**
```bash
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu
```

**For GPU (CUDA):**
```bash
pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

#### 2. **XGBoost Installation Fails**

Try installing from conda or use a pre-built wheel:
```bash
pip install xgboost --upgrade
```

#### 3. **SHAP Installation Issues**

SHAP can have dependency conflicts. If issues occur:
```bash
pip install shap --no-deps
pip install numpy scipy scikit-learn pandas matplotlib
```

#### 4. **Version Conflicts**

If you encounter version conflicts:
```bash
# Create fresh virtual environment
python -m venv venv_new
source venv_new/bin/activate  # or venv_new\Scripts\activate on Windows

# Install with upgraded pip
pip install --upgrade pip
pip install -r requirements.txt
```

#### 5. **Memory Issues During Installation**

Some packages (especially PyTorch) are large. If you run out of space:
- Clear pip cache: `pip cache purge`
- Install packages one by one
- Use `--no-cache-dir` flag: `pip install --no-cache-dir -r requirements.txt`

### Platform-Specific Notes

#### Windows
- Ensure you have Visual C++ Redistributable installed
- Use PowerShell or Command Prompt (not Git Bash for activation)

#### macOS
- May need Xcode Command Line Tools: `xcode-select --install`
- For Apple Silicon (M1/M2), PyTorch should work natively

#### Linux
- May need system dependencies:
  ```bash
  sudo apt-get update
  sudo apt-get install python3-dev python3-pip build-essential
  ```

## Alternative: Using Conda

If you prefer conda:

```bash
# Create conda environment
conda create -n stock_prediction python=3.11
conda activate stock_prediction

# Install most packages via conda
conda install pandas numpy scikit-learn
conda install -c conda-forge xgboost

# Install remaining via pip
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

## Minimum System Requirements

- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 2GB for dependencies
- **Python**: 3.9+ (3.11 recommended)

## Next Steps

After installation:

1. **Train models**: `cd backend && python train_model.py`
2. **Start backend**: `uvicorn app.main:app --reload`
3. **Start frontend**: `cd frontend && streamlit run app.py`

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.


