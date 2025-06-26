# SACA - Security Audit Compliance Agent

## Quick Start

To run the application:

```bash
python run_saca.py
```

This will:
1. Check and install any missing dependencies
2. Launch the Streamlit web interface
3. Open the application in your browser

## Alternative Launch Methods

```bash
# Direct streamlit launch
streamlit run saca11_modular.py

# Install dependencies separately
python install_dependencies.py
```

## Requirements

- Python 3.8+
- tshark (for PCAP file processing)
- Internet connection for model downloads

## Features

- Security audit compliance analysis
- PCAP file processing and analysis
- Policy document comparison
- Multiple LLM model support (Lily Cybersecurity, Ollama, Hugging Face)
- Interactive chat interface
- Vector database for document storage

## File Structure

- `run_saca.py` - Main launcher script
- `saca11_modular.py` - Core application
- `config/` - Configuration settings
- `models/` - LLM model implementations
- `processors/` - File processing utilities
- `core/` - Main audit engine
- `ui/` - Streamlit interface
- `embeddings/` - Custom embeddings
- `utils/` - Utility functions
- `tests/` - Unit tests

## Support

Check the logs in `saca11.log` for troubleshooting information.