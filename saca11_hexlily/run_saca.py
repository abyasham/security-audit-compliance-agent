#!/usr/bin/env python3
"""
Simple startup script for SACA
This script provides an easy way to run the application
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    return True

def check_dependencies():
    """Check if critical dependencies are installed"""
    missing = []
    # Map import names to package names for pip install
    deps_mapping = {
        "streamlit": "streamlit",
        "InstructorEmbedding": "InstructorEmbedding",
        "torch": "torch",
        "transformers": "transformers",
        "langchain": "langchain",
        "langchain_community": "langchain-community",
        "chromadb": "chromadb",
        "PyPDF2": "PyPDF2",
        "requests": "requests"
    }
    
    for import_name, package_name in deps_mapping.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    return missing

def install_dependencies(missing_deps):
    """Install missing dependencies"""
    print(f"Missing dependencies: {missing_deps}")
    print("Installing dependencies...")
    
    try:
        # Install dependencies one by one for better error reporting
        for dep in missing_deps:
            print(f"Installing {dep}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", dep
            ])
        print("✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during installation: {e}")
        return False

def main():
    """Main function to run SACA"""
    print("🔒 SACA - Security Audit Compliance Agent")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check if we're in the right directory
    if not Path("saca11_modular.py").exists():
        print("❌ Error: saca11_modular.py not found in current directory")
        print("Please run this script from the SACA project directory")
        return 1
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"⚠️  Missing dependencies: {missing_deps}")
        response = input("Install missing dependencies? (y/n): ").lower().strip()
        if response.startswith('y'):
            if not install_dependencies(missing_deps):
                print("❌ Failed to install dependencies")
                print("Please try manually: pip install -r requirements.txt")
                print("Or run: python install_dependencies.py")
                return 1
            print("🔄 Rechecking dependencies after installation...")
            # Recheck dependencies after installation
            remaining_deps = check_dependencies()
            if remaining_deps:
                print(f"❌ Some dependencies still missing: {remaining_deps}")
                print("Please install manually or check for compatibility issues")
                return 1
        else:
            print("❌ Missing dependencies are required to run SACA")
            print("Please install them using one of these methods:")
            print("  1. pip install -r requirements.txt")
            print("  2. python install_dependencies.py")
            return 1
    else:
        print("✅ All dependencies are available")
    
    # Run the application
    print("🚀 Starting SACA...")
    print("The application will open in your web browser")
    print("If the browser doesn't open automatically, go to: http://localhost:8501")
    print("Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        # Run streamlit with additional options for better reliability
        cmd = [
            sys.executable, "-m", "streamlit", "run", "saca11_modular.py",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ]
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 SACA stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        print("Try running manually: streamlit run saca11_modular.py")
        return 1
    except FileNotFoundError:
        print("❌ Streamlit not found. Please ensure it's installed:")
        print("pip install streamlit")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error running SACA: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())