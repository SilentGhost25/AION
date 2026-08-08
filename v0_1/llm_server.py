"""
AION Server LLM Wrapper
========================
Optimized for the production L40 server with qwen2.5:14b.
NOT loaded by default. Only activated when AION_DEVICE=server.

Usage in aion_api.py:
    from v0_1.llm_server import get_server_llm
    llm = get_server_llm()   # only if AION_DEVICE == server

This file makes ZERO changes to v0_1/llm.py.
"""

import os
from typing import Optional


def get_server_llm():
    """
    Returns an LLM caller optimized for the L40 server.
    Falls back to standard llm if not in server mode.
    """
    from core.config.production_model import get_production_model
    from v0_1.llm import RobustLLMCaller

    model = os.environ.get("AION_MODEL") or get_production_model()

    caller = RobustLLMCaller(
        primary_model = model,
        ollama_url    = "http://127.0.0.1:11434",
        timeout_sec   = 300,     # server can wait longer
    )

    # Server-optimized generation options
    caller._server_options = {
        "num_predict":    800,
        "temperature":    0.1,
        "top_p":          0.9,
        "top_k":          20,
        "repeat_penalty": 1.1,
        "num_gpu":        99,
        "num_thread":     18,
        "num_batch":      1024,  # larger batch for server GPU
        "f16_kv":         True,
    }

    return caller


def is_server_mode() -> bool:
    return os.environ.get("AION_DEVICE", "").lower() in ("server", "production")
