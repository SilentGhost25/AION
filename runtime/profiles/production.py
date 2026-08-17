# runtime/profiles/production.py
"""Production runtime profile — L40 GPU with Qwen2.5-14B."""

from typing import Optional
from runtime.profiles.base import RuntimeProfile, TimeoutBudget


class ProductionProfile(RuntimeProfile):
    """Full-power production configuration for L40 GPU servers."""

    @property
    def name(self) -> str:
        return "PRODUCTION"

    @property
    def model_name(self) -> str:
        return "qwen2.5:14b"

    @property
    def quantization(self) -> Optional[str]:
        return None  # Full precision on L40

    @property
    def backend(self) -> str:
        return "ollama"

    @property
    def device(self) -> str:
        return "CUDA"

    @property
    def concurrency(self) -> int:
        return 3

    @property
    def retrieval_strategy(self) -> str:
        return "semantic"

    @property
    def retrieval_top_k(self) -> int:
        return 8

    @property
    def context_length(self) -> int:
        return 8192

    @property
    def max_retries(self) -> int:
        return 3

    @property
    def caching_enabled(self) -> bool:
        return False

    @property
    def timeout_budget(self) -> TimeoutBudget:
        # Production has generous budgets
        return TimeoutBudget(
            hard_deadline=1200.0,
            target=900.0,
            dataset_discovery=30.0,
            extraction=180.0,
            indexing=60.0,
            planning=30.0,
            generation=600.0,
            validation=120.0,
            assembly_export=60.0,
            per_slot=90.0,
        )

    @property
    def memory_governor_enabled(self) -> bool:
        return False  # L40 has dedicated VRAM
