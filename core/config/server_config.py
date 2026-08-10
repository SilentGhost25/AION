"""
AION Production Server Configuration
====================================
Server deployment settings for Ollama parallel limits, model fallback policies, and device profile.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServerConfig:
    device: str = os.environ.get("AION_DEVICE", "server")
    model: str = os.environ.get("AION_MODEL", "qwen2.5:14b")
    allow_model_fallback: bool = os.environ.get("AION_ALLOW_MODEL_FALLBACK", "false").lower() == "true"
    ollama_num_parallel: int = int(os.environ.get("OLLAMA_NUM_PARALLEL", "1"))
    ollama_max_loaded_models: int = int(os.environ.get("OLLAMA_MAX_LOADED_MODELS", "1"))


SERVER_CONFIG = ServerConfig()
