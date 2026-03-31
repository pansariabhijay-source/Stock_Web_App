# Environment Setup — Windows + GTX 1650 Ti

Step-by-step setup. Every command is copy-paste ready.
Run these in **Anaconda Prompt** (not regular cmd or PowerShell).

---

## 1. Install Miniconda (if not already installed)

Download from: https://docs.conda.io/en/latest/miniconda.html
Choose: Python 3.11, Windows 64-bit

---

## 2. Create the conda environment

```bash
conda create -n alphastock python=3.11 -y
conda activate alphastock
```

---

## 3. Install PyTorch with CUDA 11.8 support

Your GTX 1650 Ti supports CUDA 11.8. Install PyTorch BEFORE requirements.txt
because pip's torch install can conflict with conda's CUDA packages.

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU is detected:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
Expected: `True` and `NVIDIA GeForce GTX 1650 Ti`

---

## 4. Install all other dependencies

```bash
pip install -r requirements.txt
```

If pandas-ta gives errors, try:
```bash
pip install pandas-ta --no-deps
pip install pandas numpy
```

---

## 5. Create your .env file

Create a file called `.env` inside the `alpha_stock/` folder:

```
NEWS_API_KEY=get_free_key_from_newsapi.org
HF_TOKEN=get_free_token_from_huggingface.co/settings/tokens
HF_REPO_ID=your-hf-username/alpha-stock-models
```

Get your free NewsAPI key at: https://newsapi.org (500 requests/day free)
Get your free HF token at: https://huggingface.co/settings/tokens

---

## 6. Run the data ingestion (first time only — takes ~10 minutes)

```bash
cd alpha_stock
python -m data_pipeline.ingestion
```

This downloads 10 years of data for all 50 Nifty stocks and saves to disk.
You only need to run this once. After that, use incremental updates.

---

## 7. Verify everything works

```bash
python -m features.technical
```

Should print feature count (~150) and a sample DataFrame tail.

---

## Google Colab Setup (for TFT/PatchTST training)

When we get to Phase 3, use this at the top of your Colab notebook:

```python
!pip install pytorch-forecasting pytorch-lightning huggingface-hub pandas-ta

from google.colab import drive
drive.mount('/content/drive')

# Pull feature data from your local machine (upload to Drive first)
import pandas as pd
features = pd.read_parquet('/content/drive/MyDrive/alpha_stock/features/...')
```

---

## Render Deployment (Phase 5)

Render will use the `Dockerfile` we'll write in Phase 5.
The free tier gives 512MB RAM — enough for inference only (not training).
Model artifacts are loaded from Hugging Face Hub at startup.
