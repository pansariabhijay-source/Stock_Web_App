"""
Automated installation script for backend and frontend dependencies.
"""
import subprocess
import sys
from pathlib import Path

def install_requirements(file_path: Path, name: str):
    """Install requirements from a file."""
    print(f"\n{'='*60}")
    print(f"Installing {name} dependencies...")
    print(f"{'='*60}")
    
    if not file_path.exists():
        print(f"❌ Error: {file_path} not found!")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(file_path)],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ {name} dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing {name} dependencies:")
        print(e.stderr)
        return False

def main():
    """Main installation function."""
    print("="*60)
    print("Stock Prediction Application - Dependency Installer")
    print("="*60)
    
    # Get project root
    root = Path(__file__).parent
    
    # Install backend dependencies
    backend_req = root / "backend" / "requirements.txt"
    backend_success = install_requirements(backend_req, "Backend")
    
    # Install frontend dependencies
    frontend_req = root / "frontend" / "requirements.txt"
    frontend_success = install_requirements(frontend_req, "Frontend")
    
    # Summary
    print("\n" + "="*60)
    print("Installation Summary")
    print("="*60)
    
    if backend_success and frontend_success:
        print("✅ All dependencies installed successfully!")
        print("\nNext steps:")
        print("1. Train models: cd backend && python train_model.py")
        print("2. Start backend: cd backend && uvicorn app.main:app --reload")
        print("3. Start frontend: cd frontend && streamlit run app.py")
    else:
        print("❌ Some dependencies failed to install.")
        print("Please check the error messages above and install manually.")
        if not backend_success:
            print("\nTo install backend manually:")
            print("  cd backend && pip install -r requirements.txt")
        if not frontend_success:
            print("\nTo install frontend manually:")
            print("  cd frontend && pip install -r requirements.txt")
    
    print("="*60)

if __name__ == "__main__":
    main()


