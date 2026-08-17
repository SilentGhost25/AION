# runtime/inference/llama_cpp_backend.py
"""llama.cpp inference backend — supports Vulkan and SYCL GPU acceleration.

Requires `llama-cpp-python` installed with the appropriate GPU backend.
Falls back to CPU if no GPU acceleration is available.
"""

import time
from typing import Optional

from runtime.inference.base import InferenceBackend, InferenceResult


class LlamaCppBackend(InferenceBackend):
    """llama.cpp inference backend via llama-cpp-python bindings."""

    def __init__(self):
        self._llm = None
        self._model_name: str = ""
        self._available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "llamacpp"

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import llama_cpp
            self._available = True
        except ImportError:
            self._available = False
        return self._available

    def load_model(
        self,
        model_name: str,
        quantization: Optional[str] = None,
        device: str = "GPU",
    ) -> None:
        """Load a GGUF model file.

        model_name should be a path to a .gguf file.
        """
        if not self.is_available():
            raise RuntimeError("llama-cpp-python is not installed")

        from llama_cpp import Llama

        # Determine GPU layers based on device preference
        n_gpu_layers = -1 if device.upper() in ("GPU", "VULKAN", "SYCL") else 0

        self._model_name = model_name
        self._llm = Llama(
            model_path=model_name,
            n_ctx=4096,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> InferenceResult:
        if self._llm is None:
            raise RuntimeError("No model loaded.  Call load_model() first.")

        t0 = time.perf_counter()
        output = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            echo=False,
        )
        elapsed = time.perf_counter() - t0

        text = output["choices"][0]["text"] if output.get("choices") else ""
        tokens = output.get("usage", {}).get("completion_tokens", len(text.split()))

        return InferenceResult(
            text=text,
            tokens_generated=tokens,
            latency_seconds=elapsed,
            first_token_seconds=0.0,  # llama-cpp-python doesn't expose this
            backend_name=self.name,
            model_name=self._model_name,
        )

    def unload_model(self) -> None:
        self._llm = None
        import gc
        gc.collect()

    def get_memory_usage_mb(self) -> float:
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
