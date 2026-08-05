"""
AION Unified LLM Interface — Robust LLM Caller
Features:
- Hard thread-level timeout per call
- Fallback model chain (qwen2.5:7b -> qwen2.5:3b)
- Stream keepalive callback during long calls
- Pipeline abort on consecutive failures
"""

from __future__ import annotations

import os
import json
import time
import queue
import threading
import requests
from typing import Optional, Callable


class RobustLLMCaller:
    """
    LLM caller with:
    - Hard timeout per call
    - Automatic fallback model chain
    - Stream keepalive during long calls
    - Pipeline abort on repeated failure
    """

    def __init__(
        self,
        primary_model:  Optional[str] = None,
        fallback_models: Optional[list[str]] = None,
        timeout_sec:    int = 300,
        max_retries:    int = 2,
        ollama_url:     str = "http://localhost:11434",
    ):
        self.primary_model  = primary_model or os.environ.get("AION_MODEL", "qwen2.5:7b")
        self.fallback_models = fallback_models or [
            "qwen2.5:3b",
        ]
        self.timeout_sec    = timeout_sec
        self.max_retries    = max_retries
        self.ollama_url     = ollama_url.rstrip("/")
        self._consecutive_failures = 0
        self.MAX_CONSECUTIVE_FAILURES = 3
        print(f"[LLM] RobustLLMCaller initialized — primary: {self.primary_model}")

    def call(
        self,
        prompt:     str,
        max_tokens: int = 512,
        stream_fn:  Optional[Callable[[dict], None]] = None,
    ) -> Optional[str]:
        """
        Call LLM with hard timeout and fallback.
        stream_fn: optional callback for SSE keepalive
        """
        models_to_try = [self.primary_model] + [
            m for m in self.fallback_models if m != self.primary_model
        ]

        for model in models_to_try:
            for attempt in range(self.max_retries):
                if stream_fn:
                    stream_fn({
                        "type":    "keepalive",
                        "message": f"Calling {model} (attempt {attempt+1})...",
                    })

                result = self._call_with_timeout(model, prompt, max_tokens, self.timeout_sec)

                if result is not None:
                    self._consecutive_failures = 0
                    return result

                print(f"[LLM] {model} failed (attempt {attempt+1}/{self.max_retries})")
                time.sleep(1)

        self._consecutive_failures += 1
        if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            raise RuntimeError(
                f"[LLM] ABORT: {self._consecutive_failures} consecutive LLM failures. "
                f"Is Ollama running? Ensure: ollama serve"
            )

        return None

    def _call_with_timeout(
        self,
        model:      str,
        prompt:     str,
        max_tokens: int,
        timeout:    int,
    ) -> Optional[str]:
        result_queue: queue.Queue = queue.Queue()

        def _worker():
            try:
                r = requests.post(
                    f"{self.ollama_url}/api/chat",
                    json={
                        "model":  model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {
                            "num_predict":    min(max_tokens, 1024),
                            "temperature":    0.1,
                            "top_p":          0.9,
                            "top_k":          40,
                            "repeat_penalty": 1.1,
                            "num_thread":     14,
                            "num_batch":      512,
                            "num_gpu":        1,
                            "low_vram":       False,
                            "f16_kv":         True,
                        },
                    },
                    timeout=timeout,
                )
                if r.status_code == 200:
                    data = r.json()
                    content = data.get("message", {}).get("content", "").strip()
                    if not content:
                        content = data.get("response", "").strip()
                    result_queue.put(content if content else None)
                else:
                    result_queue.put(None)

            except requests.Timeout:
                print(f"[LLM] {model} timed out after {timeout}s")
                result_queue.put(None)
            except Exception as e:
                print(f"[LLM] {model} error: {e}")
                result_queue.put(None)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        thread.join(timeout=timeout + 5)

        if thread.is_alive():
            print(f"[LLM] {model} thread hung — moving to fallback")
            return None

        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return None


class AIONLLM:
    """Wrapper around RobustLLMCaller providing generate() interface."""

    def __init__(self, model: Optional[str] = None, host: str = "http://localhost:11434"):
        self.preferred_model = model or os.environ.get("AION_MODEL", "qwen2.5:7b")
        self.caller = RobustLLMCaller(primary_model=self.preferred_model, ollama_url=host)

    def generate(
        self,
        prompt:      str,
        system:      Optional[str] = None,
        temperature: float = 0.75,
        options:     Optional[dict] = None,
        stream_fn:   Optional[Callable[[dict], None]] = None,
    ) -> str:
        max_tokens = 512
        if options and "num_predict" in options:
            max_tokens = options["num_predict"]

        full_prompt = f"System: {system}\n\nUser: {prompt}" if system else prompt
        res = self.caller.call(full_prompt, max_tokens=max_tokens, stream_fn=stream_fn)
        return res or ""


_default_llm: Optional[AIONLLM] = None


def get_llm(model: Optional[str] = None) -> AIONLLM:
    global _default_llm
    if _default_llm is None or (model and _default_llm.preferred_model != model):
        _default_llm = AIONLLM(model=model)
    return _default_llm

