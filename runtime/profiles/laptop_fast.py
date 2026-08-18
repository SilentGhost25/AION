# runtime/profiles/laptop_fast.py
"""LaptopFast runtime profile — benchmark-driven configuration for Intel Core Ultra + Arc iGPU."""

import json
import os
from pathlib import Path
from typing import Optional

from runtime.profiles.base import RuntimeProfile, TimeoutBudget

# Default cache directory for persisted benchmark results
_CACHE_DIR = Path(".aion_cache")
_PROFILE_JSON = _CACHE_DIR / "runtime_profile.json"


class LaptopFastProfile(RuntimeProfile):
    """Optimised profile for Lenovo Core Ultra 5 / 16 GB / Intel Arc iGPU.

    On first run the benchmark populates `.aion_cache/runtime_profile.json`.
    Subsequent instantiations read the cached winner so benchmarking is skipped.
    Before the benchmark has run, conservative defaults are used.
    """

    def __init__(self):
        self._cached: Optional[dict] = None
        if _PROFILE_JSON.exists():
            try:
                with open(_PROFILE_JSON, "r", encoding="utf-8") as f:
                    self._cached = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._cached = None

    # -- Identity ------------------------------------------------------

    @property
    def name(self) -> str:
        return "LAPTOP_FAST"

    # -- Model / Backend (benchmark-driven) ----------------------------

    @property
    def model_name(self) -> str:
        if self._cached:
            return self._cached.get("model", "qwen2.5:3b")
        return "qwen2.5:3b"  # Conservative pre-benchmark default

    @property
    def quantization(self) -> Optional[str]:
        if self._cached:
            return self._cached.get("quantization", "INT4")
        return "INT4"

    @property
    def backend(self) -> str:
        if self._cached:
            return self._cached.get("backend", "ollama")
        return "ollama"  # Safest pre-benchmark default

    @property
    def device(self) -> str:
        if self._cached:
            return self._cached.get("device", "GPU")
        return "GPU"

    # -- Concurrency ---------------------------------------------------

    @property
    def concurrency(self) -> int:
        return 1  # Always single-worker on laptop

    # -- Retrieval -----------------------------------------------------

    @property
    def retrieval_strategy(self) -> str:
        return "bm25"

    @property
    def retrieval_top_k(self) -> int:
        if self._cached:
            return self._cached.get("top_k", 4)
        return 4

    # -- Context -------------------------------------------------------

    @property
    def context_length(self) -> int:
        if self._cached:
            return self._cached.get("context_length", 4096)
        return 4096

    # -- Recovery ------------------------------------------------------

    @property
    def max_retries(self) -> int:
        return 2  # Bounded retries on laptop

    # -- Caching -------------------------------------------------------

    @property
    def caching_enabled(self) -> bool:
        return True  # Always cache on laptop

    # -- Timeouts ------------------------------------------------------

    @property
    def timeout_budget(self) -> TimeoutBudget:
        return TimeoutBudget(
            hard_deadline=600.0,
            target=540.0,
            dataset_discovery=5.0,
            extraction=60.0,
            indexing=15.0,
            planning=5.0,
            generation=390.0,
            validation=40.0,
            assembly_export=25.0,
            per_slot=480.0,
        )

    # -- Memory Governor -----------------------------------------------

    @property
    def memory_governor_enabled(self) -> bool:
        return True  # Shared-memory iGPU requires monitoring

    # -- Persistence ---------------------------------------------------

    def persist_benchmark_result(self, result: dict) -> None:
        """Save benchmark winner to disk so future runs skip benchmarking."""
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(_PROFILE_JSON, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        self._cached = result
