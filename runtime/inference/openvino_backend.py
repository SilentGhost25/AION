# runtime/inference/openvino_backend.py
"""OpenVINO inference backend for Intel Arc iGPU acceleration.

Attempts to use `openvino_genai` for high-level pipeline inference,
falling back to core `openvino` runtime if the genai package is absent.
"""

import time
from typing import Optional

from runtime.inference.base import InferenceBackend, InferenceResult


class OpenVINOBackend(InferenceBackend):
    """OpenVINO GPU/CPU inference backend."""

    def __init__(self):
        self._pipeline = None
        self._tokenizer = None
        self._model_name: str = ""
        self._device: str = "GPU"
        self._ov_available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "openvino"

    def is_available(self) -> bool:
        if self._ov_available is not None:
            return self._ov_available
        try:
            import openvino
            self._ov_available = True
        except ImportError:
            self._ov_available = False
        return self._ov_available

    def load_model(
        self,
        model_name: str,
        quantization: Optional[str] = None,
        device: str = "GPU",
    ) -> None:
        """Load a Qwen model via OpenVINO GenAI pipeline.

        model_name should be a HuggingFace model ID or a local path to an
        OpenVINO IR model directory.
        """
        if not self.is_available():
            raise RuntimeError("OpenVINO is not installed")

        self._device = device
        self._model_name = model_name

        try:
            import openvino_genai as ov_genai

            self._pipeline = ov_genai.LLMPipeline(model_name, device)
            self._tokenizer = None  # GenAI pipeline handles tokenisation
        except ImportError:
            # Fallback: signal that genai is unavailable
            raise RuntimeError(
                "openvino_genai package is required for LLM inference.  "
                "Install via: pip install openvino-genai"
            )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> InferenceResult:
        if self._pipeline is None:
            raise RuntimeError("No model loaded.  Call load_model() first.")

        import openvino_genai as ov_genai

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = max_tokens
        config.temperature = temperature
        config.do_sample = temperature > 0

        t0 = time.perf_counter()
        first_token_time = 0.0
        tokens_counted = 0

        # Use streamer to capture first-token latency
        output_text = ""

        def _streamer(token_text: str) -> bool:
            nonlocal first_token_time, tokens_counted, output_text
            if tokens_counted == 0:
                first_token_time = time.perf_counter() - t0
            tokens_counted += 1
            output_text += token_text
            return False  # False = continue generation

        try:
            self._pipeline.generate(prompt, config, _streamer)
        except TypeError:
            # Some openvino_genai versions use different API
            result = self._pipeline.generate(prompt, config)
            output_text = str(result)
            tokens_counted = len(output_text.split())
            first_token_time = 0.0

        elapsed = time.perf_counter() - t0

        return InferenceResult(
            text=output_text,
            tokens_generated=tokens_counted,
            latency_seconds=elapsed,
            first_token_seconds=first_token_time,
            backend_name=self.name,
            model_name=self._model_name,
        )

    def unload_model(self) -> None:
        self._pipeline = None
        self._tokenizer = None
        import gc
        gc.collect()

    def get_memory_usage_mb(self) -> float:
        """Approximate memory usage — OpenVINO doesn't expose this directly."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
