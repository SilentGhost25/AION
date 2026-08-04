"""
AION v0.1 Pipeline Package
"""

from .generator import generate, generate_turbo

def __getattr__(name):
    if name == "run_pipeline":
        from .main import run_pipeline
        return run_pipeline
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ["generate", "generate_turbo", "run_pipeline"]
