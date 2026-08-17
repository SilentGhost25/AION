# runtime/inference/ollama_backend.py
"""Ollama inference backend — wraps the local Ollama HTTP API.

This is the most portable backend: if Ollama is running locally and has
the model pulled, it works on any OS without additional dependencies.
"""

import json
import time
import urllib.request
import urllib.error
from typing import Optional

from runtime.inference.base import InferenceBackend, InferenceResult


class OllamaBackend(InferenceBackend):
    """Local Ollama API inference backend."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self._base_url = base_url.rstrip("/")
        self._model_name: str = ""
        self._loaded: bool = False

    @property
    def name(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        """Check if Ollama is running and responding."""
        try:
            req = urllib.request.Request(f"{self._base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def load_model(
        self,
        model_name: str,
        quantization: Optional[str] = None,
        device: str = "GPU",
    ) -> None:
        """Set model name and verify it exists in Ollama."""
        if not self.is_available():
            raise RuntimeError("Ollama is not running")

        self._model_name = model_name

        # Verify model is available
        try:
            req = urllib.request.Request(f"{self._base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                # Ollama model names may include `:latest`
                found = any(
                    model_name in m or m.startswith(model_name)
                    for m in models
                )
                if not found:
                    raise RuntimeError(
                        f"Model '{model_name}' not found in Ollama.  "
                        f"Available: {models}.  Run: ollama pull {model_name}"
                    )
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Cannot reach Ollama: {exc}") from exc

        self._loaded = True

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> InferenceResult:
        if not self._loaded:
            raise RuntimeError("No model loaded.  Call load_model() first.")

        payload = json.dumps({
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama inference failed: {exc}") from exc

        elapsed = time.perf_counter() - t0

        text = raw.get("response", "")
        # Ollama reports eval_count as tokens generated
        tokens = raw.get("eval_count", len(text.split()))

        # Compute first-token latency from Ollama timing if available
        prompt_eval_ns = raw.get("prompt_eval_duration", 0)
        first_token = prompt_eval_ns / 1e9 if prompt_eval_ns else 0.0

        return InferenceResult(
            text=text,
            tokens_generated=tokens,
            latency_seconds=elapsed,
            first_token_seconds=first_token,
            backend_name=self.name,
            model_name=self._model_name,
        )

    def unload_model(self) -> None:
        self._loaded = False
        self._model_name = ""

    def get_memory_usage_mb(self) -> float:
        """Query Ollama's reported VRAM usage if available."""
        try:
            req = urllib.request.Request(f"{self._base_url}/api/ps")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models", [])
                total = sum(m.get("size_vram", 0) for m in models)
                return total / (1024 * 1024)
        except Exception:
            return 0.0
