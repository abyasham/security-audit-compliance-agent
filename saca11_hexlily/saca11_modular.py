"""
SACA - Security Audit Compliance Agent (Modular Version)
Main entry point for the refactored modular application

This is the new modular version of the original saca11_hexlily.py file.
The application has been broken down into logical modules for better maintainability:

- config/settings.py: Configuration and settings
- models/llm_models.py: LLM model implementations
- processors/file_processors.py: File processing utilities
- core/audit_engine.py: Main audit logic and Saca11 class
- ui/streamlit_ui.py: Streamlit user interface
- embeddings/embeddings.py: Custom embedding implementations
- utils/utils.py: Utility functions including text splitting

Usage:
    streamlit run saca11_modular.py
"""

import sys
import os
import logging
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import configuration first
from config.settings import LOGGING_CONFIG, validate_config

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    try:
        logging.basicConfig(
            level=getattr(logging, LOGGING_CONFIG["level"]),
            format=LOGGING_CONFIG["format"],
            handlers=[
                logging.FileHandler(LOGGING_CONFIG["log_file"]),
                logging.StreamHandler()
            ]
        )
        logging.info("SACA Modular Application Starting...")
    except Exception as e:
        print(f"Warning: Could not setup logging: {e}")
        logging.basicConfig(level=logging.INFO)


def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = [
        'streamlit',
        'torch',
        'transformers',
        'langchain',
        'langchain_community',
        'chromadb',
        'InstructorEmbedding',
        'PyPDF2',
        'requests'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logging.error(f"Missing required modules: {missing_modules}")
        print(f"Error: Missing required modules: {missing_modules}")
        print("Please install them using: pip install " + " ".join(missing_modules))
        return False
    
    return True


def check_external_tools():
    """Check if external tools like tshark are available"""
    import subprocess
    
    try:
        # Check if tshark is available
        result = subprocess.run(['tshark', '-v'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        if result.returncode == 0:
            logging.info("tshark is available for PCAP processing")
        else:
            logging.warning("tshark not found - PCAP processing may not work")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logging.warning("tshark not found - PCAP processing may not work")
        print("Warning: tshark not found. Install Wireshark to enable PCAP file processing.")


def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        "persist",
        "models",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created directory: {directory}")


def main():
    """Main entry point for the application"""
    try:
        # Setup logging
        setup_logging()
        
        # Validate configuration
        validate_config()
        logging.info("Configuration validated successfully")
        
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)
        
        # Check external tools
        check_external_tools()
        
        # Create necessary directories
        create_directories()
        
        # Import and run the Streamlit UI
        from ui.streamlit_ui import main_ui
        
        logging.info("Starting Streamlit UI...")
        main_ui()
        
    except KeyboardInterrupt:
        logging.info("Application interrupted by user")
        print("\nApplication stopped by user")
    except Exception as e:
        logging.error(f"Fatal error in main application: {str(e)}")
        print(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()