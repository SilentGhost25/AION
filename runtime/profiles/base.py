# runtime/profiles/base.py
"""Abstract base class for all AION runtime profiles.

A RuntimeProfile controls execution parameters — model selection,
concurrency, retrieval strategy, caching, timeouts — but NEVER
modifies the core AION contracts (CO, Bloom, marks, module locking,
evidence binding, validators, or ExportGate).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TimeoutBudget:
    """Breakdown of the global time limit into phase budgets (seconds)."""

    hard_deadline: float = 600.0
    target: float = 540.0
    dataset_discovery: float = 10.0
    extraction: float = 60.0
    indexing: float = 20.0
    planning: float = 10.0
    generation: float = 360.0
    validation: float = 60.0
    assembly_export: float = 30.0
    per_slot: float = 120.0


class RuntimeProfile(ABC):
    """Abstract base defining the parameters every profile must specify.

    Subclasses set concrete values.  The AION core pipeline reads these
    values at runtime but never modifies them.
    """

    # -- Identity ------------------------------------------------------
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable profile name, e.g. 'PRODUCTION'."""

    # -- Model / Backend -----------------------------------------------
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Qwen model identifier used for inference."""

    @property
    @abstractmethod
    def quantization(self) -> Optional[str]:
        """Quantization level, e.g. 'INT4', or None for full precision."""

    @property
    @abstractmethod
    def backend(self) -> str:
        """Inference backend: 'openvino', 'llamacpp', 'ollama', 'cuda'."""

    @property
    @abstractmethod
    def device(self) -> str:
        """Target device: 'GPU', 'CPU', 'CUDA'."""

    # -- Concurrency ---------------------------------------------------
    @property
    @abstractmethod
    def concurrency(self) -> int:
        """Maximum parallel Qwen inference workers."""

    # -- Retrieval -----------------------------------------------------
    @property
    @abstractmethod
    def retrieval_strategy(self) -> str:
        """'semantic', 'bm25', or 'hybrid'."""

    @property
    @abstractmethod
    def retrieval_top_k(self) -> int:
        """Number of evidence chunks to retrieve per query."""

    # -- Context -------------------------------------------------------
    @property
    @abstractmethod
    def context_length(self) -> int:
        """Maximum context window tokens for the LLM."""

    # -- Recovery ------------------------------------------------------
    @property
    @abstractmethod
    def max_retries(self) -> int:
        """Maximum retry attempts per slot on validation failure."""

    # -- Caching -------------------------------------------------------
    @property
    @abstractmethod
    def caching_enabled(self) -> bool:
        """Whether extraction/index caching is enabled."""

    # -- Timeouts ------------------------------------------------------
    @property
    @abstractmethod
    def timeout_budget(self) -> TimeoutBudget:
        """Phase-aware timeout budget."""

    # -- Memory Governor -----------------------------------------------
    @property
    @abstractmethod
    def memory_governor_enabled(self) -> bool:
        """Whether the dynamic memory governor is active."""

    # -- Utilities -----------------------------------------------------
    def summary(self) -> dict:
        """Return a dict summarising the profile for logging/display."""
        return {
            "profile": self.name,
            "model": self.model_name,
            "quantization": self.quantization,
            "backend": self.backend,
            "device": self.device,
            "concurrency": self.concurrency,
            "retrieval": self.retrieval_strategy,
            "top_k": self.retrieval_top_k,
            "context_length": self.context_length,
            "max_retries": self.max_retries,
            "caching": self.caching_enabled,
            "memory_governor": self.memory_governor_enabled,
            "hard_deadline": self.timeout_budget.hard_deadline,
            "target": self.timeout_budget.target,
        }
