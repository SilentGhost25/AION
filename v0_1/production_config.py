"""
AION Production Server Configuration & Environment Policy
=========================================================
Freezes single generation model (qwen2.5:14b), concurrency (1), fail-closed thresholds,
and server parameters.
"""

import os

SERVER_CONFIG = {
    "device": os.getenv("AION_DEVICE", "server"),
    "model": os.getenv("AION_MODEL", "qwen2.5:14b"),
    "port": int(os.getenv("AION_PORT", "8100")),
    "max_parallel": 1,
    "max_loaded_models": 1,
    "unified_pipeline": True,
    "strict_grounding": True,
    "strict_validation": True,
    "strict_rendering": True,
    "allow_external_knowledge": False,
    "allow_unsupported_synthesis": False,
    "allow_model_fallback": False,
    "max_repair_passes": 1,
    "retrieval_candidates": 10,
    "retrieval_final": 3,
    "min_extraction_confidence": 0.70,
    "min_grounding_score": 0.80,
    "vre_enabled": True,
    "auto_healer_enabled": True,
}


def get_production_server_profile() -> dict:
    """Returns the locked production server configuration dict."""
    return dict(SERVER_CONFIG)
