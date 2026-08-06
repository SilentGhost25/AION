"""
AION Unified LLM Interface — Robust LLM Caller
Single Production Model: qwen2.5:7b (core/config/production_model.py)
Policy: No silent fallback, no automatic downgrade. Fail loud on production model failure.
Features:
- Hard thread-level timeout per call
- Centralized production model import
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

try:
    from core.config.production_model import PRODUCTION_MODEL, get_production_model
except ImportError:
    PRODUCTION_MODEL = "qwen2.5:7b"
    def get_production_model():
        return os.environ.get("AION_MODEL", PRODUCTION_MODEL)


class RobustLLMCaller:
    """
    LLM caller with:
    - Hard timeout per call
    - Single production model (no silent downgrade)
    - Optional explicit fallback only if caller provides allow_fallback=True
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
        allow_fallback: bool = False,
    ):
        # Enforce production model as single source of truth
        requested = primary_model or os.environ.get("AION_MODEL", PRODUCTION_MODEL)
        if requested != PRODUCTION_MODEL:
            # Fail loud — do not silently switch. Caller must explicitly allow fallback.
            if not allow_fallback:
                print(f"[LLM] WARNING: Requested model '{requested}' differs from production '{PRODUCTION_MODEL}'. "
                      f"Using production model. Pass allow_fallback=True to override.")
                requested = PRODUCTION_MODEL
        self.primary_model  = get_production_model() if requested == PRODUCTION_MODEL else requested
        # By default NO fallback. Silent downgrade is disabled per AION Development Context.
        if fallback_models is None:
            self.fallback_models = []  # No silent fallback
        else:
            self.fallback_models = [m for m in fallback_models if m != self.primary_model]
        self.allow_fallback = allow_fallback
        self.timeout_sec    = timeout_sec
        self.max_retries    = max_retries
        self.ollama_url     = ollama_url.rstrip("/")
        self._consecutive_failures = 0
        self.MAX_CONSECUTIVE_FAILURES = 3
        print(f"[LLM] RobustLLMCaller initialized — primary: {self.primary_model} "
              f"(fallback={'enabled' if self.allow_fallback else 'DISABLED'})")

    def call(
        self,
        prompt:     str,
        max_tokens: int = 512,
        stream_fn:  Optional[Callable[[dict], None]] = None,
    ) -> Optional[str]:
        """
        Call LLM with hard timeout.
        Only production model is tried unless allow_fallback was explicitly enabled.
        """
        models_to_try = [self.primary_model]
        if self.allow_fallback and self.fallback_models:
            models_to_try += self.fallback_models

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
                f"Is Ollama running? Ensure: ollama serve && ollama pull {PRODUCTION_MODEL}"
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
                            "temperature":    0.3,
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
                    print(f"[LLM] HTTP {r.status_code}: {r.text[:200]}")
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
            print(f"[LLM] {model} thread hung — no fallback (production model only)")
            return None

        try:
            return result_queue.get_nowait()
        except queue.Empty:
            return None


class AIONLLM:
    """Wrapper around RobustLLMCaller providing generate() interface."""

    def __init__(self, model: Optional[str] = None, host: str = "http://localhost:11434"):
        # Enforce centralized production model
        requested = model or os.environ.get("AION_MODEL", PRODUCTION_MODEL)
        self.preferred_model = get_production_model() if requested in (None, PRODUCTION_MODEL) else requested
        if self.preferred_model != PRODUCTION_MODEL:
            print(f"[LLM] AIONLLM using requested '{self.preferred_model}' (production is '{PRODUCTION_MODEL}')")
        self.caller = RobustLLMCaller(primary_model=self.preferred_model, ollama_url=host)

    def generate(
        self,
        prompt:      str,
        system:      Optional[str] = None,
        temperature: float = 0.30,
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
