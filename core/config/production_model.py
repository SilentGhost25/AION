"""
AION Centralized Production Model Configuration
==============================================
Single Source of Truth for the production LLM.

REQUIREMENT (AION Development Context):
- ONE production model across entire codebase
- Current target: qwen2.5:7b via Ollama
- No silent model switching
- No automatic downgrade
- All modules MUST import from here

Any file that defines its own model default is in violation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final

# ── Single Production Model ──────────────────────────────────
PRODUCTION_MODEL: Final[str] = "qwen2.5:7b"
PRODUCTION_MODEL_ALIASES: Final[tuple[str, ...]] = ("qwen2.5:7b", "qwen2.5-7b")

# Deprecated models that must NOT be used silently
DEPRECATED_MODELS: Final[tuple[str, ...]] = (
    "qwen2.5:1.5b",
    "qwen2.5:3b",
    "aion",
    "aion-exam",
    "aion-qwen",
)

# ── Ollama Configuration ─────────────────────────────────────
DEFAULT_OLLAMA_URL: Final[str] = "http://localhost:11434"
DEFAULT_OLLAMA_HOST: Final[str] = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_URL)

# Generation defaults tuned for qwen2.5:7b
DEFAULT_GENERATION_OPTIONS: Final[dict] = {
    "temperature": 0.3,       # Lower for grounding fidelity
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "num_ctx": 4096,          # 7B can handle larger context
    "num_predict": 350,
    "seed": None,
}

# ── Helper API ───────────────────────────────────────────────

def get_production_model() -> str:
    """
    Returns the single production model.
    Environment override is allowed ONLY for explicit testing;
    otherwise always returns PRODUCTION_MODEL.
    Logs a warning if env override is deprecated.
    """
    env_model = os.environ.get("AION_MODEL", "").strip()
    if env_model and env_model not in (PRODUCTION_MODEL, "") and env_model in DEPRECATED_MODELS:
        # Warn but do not silently downgrade — caller must handle explicitly
        import warnings
        warnings.warn(
            f"[AION] Deprecated model '{env_model}' requested via AION_MODEL. "
            f"Production model is '{PRODUCTION_MODEL}'. "
            f"Downgrade requires explicit --allow-deprecated flag. Using production model.",
            UserWarning,
            stacklevel=2,
        )
        return PRODUCTION_MODEL
    if env_model and env_model == PRODUCTION_MODEL:
        return env_model
    if env_model and env_model not in DEPRECATED_MODELS and env_model != PRODUCTION_MODEL:
        # Allow experimental model only if explicitly different but not deprecated — warn
        import warnings
        warnings.warn(
            f"[AION] Non-production model '{env_model}' in use. Expected '{PRODUCTION_MODEL}'.",
            UserWarning,
            stacklevel=2,
        )
        return env_model
    return PRODUCTION_MODEL


def assert_production_model(model: str) -> None:
    """Raises if model is not production model (for CI checks)."""
    if model != PRODUCTION_MODEL:
        raise AssertionError(
            f"Model violation: expected '{PRODUCTION_MODEL}', got '{model}'. "
            f"All modules must use core.config.production_model.get_production_model()."
        )


def is_deprecated_model(model: str) -> bool:
    return model in DEPRECATED_MODELS


@dataclass(frozen=True)
class ModelIdentity:
    """Immutable production model identity."""
    name: str = PRODUCTION_MODEL
    ollama_url: str = DEFAULT_OLLAMA_URL
    context_window: int = 4096
    parameters: str = "7B"
    quantization: str | None = None

    def __str__(self) -> str:
        return self.name


# Singleton for import convenience
PRODUCTION_IDENTITY = ModelIdentity()
