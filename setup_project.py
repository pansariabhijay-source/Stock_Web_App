"""
Setup script to organize project files.
Moves data files to appropriate directories and creates necessary folders.
"""
import shutil
from pathlib import Path
import os

def setup_project():
    """Organize project files into backend/frontend structure."""
    
    # Define paths
    root = Path(".")
    backend_data = root / "backend" / "data"
    backend_models = root / "backend" / "models"
    
    # Create directories
    backend_data.mkdir(parents=True, exist_ok=True)
    backend_models.mkdir(parents=True, exist_ok=True)
    
    print("Setting up project structure...")
    
    # Move data files
    data_files = [
        "RILO - Copy.csv",
        "FINAL_FEATURES_OUT.csv",
        "output.csv",
        "output_scaled.csv"
    ]
    
    for file in data_files:
        src = root / file
        if src.exists():
            dst = backend_data / file
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"Copied {file} to backend/data/")
            else:
                print(f"{file} already exists in backend/data/")
    
    # Keep notebook in root (as requested)
    print("\nProject setup complete!")
    print("\nNext steps:")
    print("1. Train models: cd backend && python train_model.py")
    print("2. Start backend: cd backend && uvicorn app.main:app --reload")
    print("3. Start frontend: cd frontend && streamlit run app.py")

if __name__ == "__main__":
    setup_project()


