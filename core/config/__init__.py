"""
AION Core Config — Centralized Configuration Exports
"""

from .production_model import (
    PRODUCTION_MODEL,
    PRODUCTION_IDENTITY,
    DEPRECATED_MODELS,
    DEFAULT_OLLAMA_URL,
    DEFAULT_GENERATION_OPTIONS,
    get_production_model,
    assert_production_model,
    is_deprecated_model,
    ModelIdentity,
)

__all__ = [
    "PRODUCTION_MODEL",
    "PRODUCTION_IDENTITY",
    "DEPRECATED_MODELS",
    "DEFAULT_OLLAMA_URL",
    "DEFAULT_GENERATION_OPTIONS",
    "get_production_model",
    "assert_production_model",
    "is_deprecated_model",
    "ModelIdentity",
]
