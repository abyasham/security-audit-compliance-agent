"""
Configuration settings for SACA
Contains default values, model configurations, and system settings
"""

import os
from typing import Dict, Any

# Default chunk settings
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K_VIOLATIONS = 3

# Model configurations
OLLAMA_CONFIG = {
    "model_name": "deepseek-r1:8b",
    "base_url": "http://localhost:11434",
    "temperature": 0.1,
    "max_tokens": 2048,
    "timeout": 300
}

HUGGINGFACE_CONFIG = {
    "model_name": "microsoft/DialoGPT-medium",
    "device_map": "auto",
    "torch_dtype": "auto",
    "trust_remote_code": True,
    "max_length": 2048,
    "temperature": 0.1,
    "do_sample": True,
    "pad_token_id": 50256
}

# Embedding model configuration
EMBEDDING_CONFIG = {
    "model_name": "hkunlp/instructor-xl",
    "device": "cuda" if os.environ.get("CUDA_AVAILABLE", "false").lower() == "true" else "cpu",
    "cache_folder": "./models/instructor_cache",
    "instruction": "Represent the cybersecurity document for retrieval: "
}

# ChromaDB configuration
CHROMADB_CONFIG = {
    "persist_directory": "./persist",
    "collection_names": {
        "logs": "logs",
        "policies": "policies"
    },
    "distance_function": "cosine"
}

# File processing settings
FILE_PROCESSING_CONFIG = {
    "max_file_size_mb": 100,
    "supported_log_formats": [".pcap", ".txt", ".log", ".json"],
    "supported_policy_formats": [".txt", ".pdf", ".json"],
    "tshark_timeout": 300,
    "encoding": "utf-8"
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "log_file": "saca11.log",
    "max_bytes": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# Streamlit UI configuration
UI_CONFIG = {
    "page_title": "SACA - Security Audit Compliance Agent",
    "page_icon": "🔒",
    "layout": "wide",
    "sidebar_width": 300,
    "max_upload_size": 200  # MB
}

# Analysis configuration
ANALYSIS_CONFIG = {
    "max_log_chunks_per_query": 5,
    "max_log_fetch_k": 10,
    "max_policy_chunks": 10,
    "max_retrieved_chunks": 15,
    "mmr_lambda": 0.5,
    "base_queries": [
        "Identify log events that potentially violate cybersecurity policies",
        "Find suspicious network activities in logs",
        "Detect unauthorized access attempts in logs",
        "Identify data exfiltration evidence in logs",
        "Find policy violations related to authentication in logs"
    ]
}

# Memory management settings
MEMORY_CONFIG = {
    "clear_cache_after_processing": True,
    "gpu_memory_fraction": 0.8,
    "cpu_fallback": True,
    "batch_size": 32
}

# Security settings
SECURITY_CONFIG = {
    "sanitize_inputs": True,
    "max_query_length": 1000,
    "allowed_file_extensions": [
        ".pcap", ".txt", ".log", ".json", ".pdf"
    ],
    "blocked_patterns": [
        r"<script.*?>.*?</script>",
        r"javascript:",
        r"vbscript:",
        r"onload=",
        r"onerror="
    ]
}


def get_config(section: str) -> Dict[str, Any]:
    """
    Get configuration for a specific section
    
    Args:
        section: Configuration section name
        
    Returns:
        Dictionary containing configuration values
    """
    config_map = {
        "ollama": OLLAMA_CONFIG,
        "huggingface": HUGGINGFACE_CONFIG,
        "embedding": EMBEDDING_CONFIG,
        "chromadb": CHROMADB_CONFIG,
        "file_processing": FILE_PROCESSING_CONFIG,
        "logging": LOGGING_CONFIG,
        "ui": UI_CONFIG,
        "analysis": ANALYSIS_CONFIG,
        "memory": MEMORY_CONFIG,
        "security": SECURITY_CONFIG
    }
    
    return config_map.get(section, {})


def get_model_config(model_type: str) -> Dict[str, Any]:
    """
    Get model-specific configuration
    
    Args:
        model_type: Type of model ("ollama" or "huggingface")
        
    Returns:
        Model configuration dictionary
    """
    if model_type.lower() == "ollama":
        return OLLAMA_CONFIG
    elif model_type.lower() == "huggingface":
        return HUGGINGFACE_CONFIG
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def update_config_from_env():
    """Update configuration values from environment variables"""
    # Update Ollama config from environment
    if os.getenv("OLLAMA_BASE_URL"):
        OLLAMA_CONFIG["base_url"] = os.getenv("OLLAMA_BASE_URL")
    
    if os.getenv("OLLAMA_MODEL"):
        OLLAMA_CONFIG["model_name"] = os.getenv("OLLAMA_MODEL")
    
    # Update HuggingFace config from environment
    if os.getenv("HF_MODEL"):
        HUGGINGFACE_CONFIG["model_name"] = os.getenv("HF_MODEL")
    
    # Update embedding config from environment
    if os.getenv("EMBEDDING_MODEL"):
        EMBEDDING_CONFIG["model_name"] = os.getenv("EMBEDDING_MODEL")
    
    # Update ChromaDB persist directory
    if os.getenv("CHROMADB_PERSIST_DIR"):
        CHROMADB_CONFIG["persist_directory"] = os.getenv("CHROMADB_PERSIST_DIR")
    
    # Update logging level
    if os.getenv("LOG_LEVEL"):
        LOGGING_CONFIG["level"] = os.getenv("LOG_LEVEL").upper()


def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Validate chunk settings
    if DEFAULT_CHUNK_SIZE <= 0:
        errors.append("DEFAULT_CHUNK_SIZE must be positive")
    
    if DEFAULT_CHUNK_OVERLAP < 0:
        errors.append("DEFAULT_CHUNK_OVERLAP must be non-negative")
    
    if DEFAULT_CHUNK_OVERLAP >= DEFAULT_CHUNK_SIZE:
        errors.append("DEFAULT_CHUNK_OVERLAP must be less than DEFAULT_CHUNK_SIZE")
    
    # Validate file size limits
    if FILE_PROCESSING_CONFIG["max_file_size_mb"] <= 0:
        errors.append("max_file_size_mb must be positive")
    
    # Validate model configurations
    if not OLLAMA_CONFIG["model_name"]:
        errors.append("Ollama model name cannot be empty")
    
    if not HUGGINGFACE_CONFIG["model_name"]:
        errors.append("HuggingFace model name cannot be empty")
    
    # Validate embedding configuration
    if not EMBEDDING_CONFIG["model_name"]:
        errors.append("Embedding model name cannot be empty")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")


# Initialize configuration on import
update_config_from_env()

# Validate configuration
try:
    validate_config()
except ValueError as e:
    print(f"Warning: {e}")


# Export commonly used values
__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP", 
    "DEFAULT_TOP_K_VIOLATIONS",
    "OLLAMA_CONFIG",
    "HUGGINGFACE_CONFIG",
    "EMBEDDING_CONFIG",
    "CHROMADB_CONFIG",
    "FILE_PROCESSING_CONFIG",
    "LOGGING_CONFIG",
    "UI_CONFIG",
    "ANALYSIS_CONFIG",
    "MEMORY_CONFIG",
    "SECURITY_CONFIG",
    "get_config",
    "get_model_config",
    "update_config_from_env",
    "validate_config"
]