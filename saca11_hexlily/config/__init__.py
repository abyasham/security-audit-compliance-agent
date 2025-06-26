"""
Configuration module for SACA
"""

from .settings import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K_VIOLATIONS,
    get_config,
    get_model_config,
    validate_config
)

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP", 
    "DEFAULT_TOP_K_VIOLATIONS",
    "get_config",
    "get_model_config",
    "validate_config"
]