# runtime/inference/base.py
"""Abstract base class for local inference backends.

Every backend must implement `generate()` which accepts a prompt and
returns the raw LLM output string.  The caller is responsible for
JSON parsing and validation — the backend only handles inference.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceResult:
    """Result of a single inference call."""

    text: str
    tokens_generated: int
    latency_seconds: float
    first_token_seconds: float
    backend_name: str
    model_name: str

    @property
    def tokens_per_second(self) -> float:
        if self.latency_seconds <= 0:
            return 0.0
        return self.tokens_generated / self.latency_seconds


class InferenceBackend(ABC):
    """Abstract base for all inference backends (OpenVINO, llama.cpp, Ollama)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier, e.g. 'openvino', 'llamacpp', 'ollama'."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend is installed and operational."""

    @abstractmethod
    def load_model(self, model_name: str, quantization: Optional[str] = None, device: str = "GPU") -> None:
        """Load a model into memory. Must be called before generate()."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.3) -> InferenceResult:
        """Run inference and return the result."""

    @abstractmethod
    def unload_model(self) -> None:
        """Release model resources."""

    @abstractmethod
    def get_memory_usage_mb(self) -> float:
        """Estimate current memory usage of the loaded model in MB."""
