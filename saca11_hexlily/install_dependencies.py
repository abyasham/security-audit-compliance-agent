#!/usr/bin/env python3
"""
Install missing dependencies for SACA
"""

import subprocess
import sys

def install_package(package):
    """Install a Python package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    """Install missing dependencies"""
    print("🔧 Installing SACA Dependencies...")
    print("=" * 40)
    
    # Critical dependencies that are often missing
    critical_packages = [
        "InstructorEmbedding",
        "sentence-transformers",
        "streamlit",
        "torch",
        "transformers",
        "langchain",
        "langchain-community", 
        "chromadb",
        "PyPDF2",
        "requests"
    ]
    
    failed_packages = []
    
    for package in critical_packages:
        print(f"Installing {package}... ", end="")
        if install_package(package):
            print("✅ Success")
        else:
            print("❌ Failed")
            failed_packages.append(package)
    
    print("=" * 40)
    
    if failed_packages:
        print(f"❌ Failed to install: {failed_packages}")
        print("Please install them manually:")
        print(f"pip install {' '.join(failed_packages)}")
        return 1
    else:
        print("🎉 All dependencies installed successfully!")
        print("You can now run: python run_saca.py")
        return 0

if __name__ == "__main__":
    sys.exit(main())