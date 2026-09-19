"""
AION Unified LLM Interface — Robust LLM Caller
Single Production Model: qwen2.5:7b (core/config/production_model.py)
Policy: No silent fallback, no automatic downgrade. Fail loud on production model failure.
Features:
- Hard thread-level timeout per call
- Centralized production model import
- Stream keepalive callback during long calls
- Pipeline abort on consecutive failures
- Runtime profile integration for LAPTOP_FAST / LAPTOP_DEMO
- Capability detection: chat / generate / none (probed via /api/show)
"""

from __future__ import annotations

import os
import json
import time
import queue
import threading
import requests
from typing import Optional, Callable, Dict

from core.config.production_model import get_production_model


# -- Model capability cache -------------------------------------------------------
# Values: "chat" | "generate" | "none"
# Populated once per process; probes are not repeated.
_MODEL_CAPABILITY: Dict[str, str] = {}
_CAPABILITY_LOCK = threading.Lock()


def get_llm_backend() -> str:
    env_backend = os.environ.get("AION_BACKEND") or os.environ.get("LLM_BACKEND") or os.environ.get("BACKEND")
    if env_backend:
        return env_backend.lower().strip()
    # Auto-detect running vLLM server on localhost:8000
    try:
        r = requests.get("http://localhost:8000/v1/models", timeout=0.3)
        if r.status_code == 200:
            return "vllm"
    except Exception:
        pass
    return "ollama"


def get_llm_host(default_ollama: str = "http://127.0.0.1:11434") -> str:
    if get_llm_backend() == "vllm":
        return (os.environ.get("AION_LLM_HOST") or os.environ.get("LLM_BASE_URL") or os.environ.get("LLM_HOST") or os.environ.get("VLLM_URL") or "http://localhost:8000").rstrip("/")
    return (os.environ.get("OLLAMA_URL") or os.environ.get("OLLAMA_HOST") or os.environ.get("AION_LLM_HOST") or default_ollama).rstrip("/")


def probe_model_capability(
    model: str,
    ollama_url: Optional[str] = None,
) -> str:
    """
    Determine which LLM API endpoint a model supports.
    Returns "vllm_chat", "chat", "generate", or "none".
    """
    if model == "AUTO":
        try:
            model = os.environ.get("AION_MODEL") or get_production_model()
        except Exception:
            model = "qwen2.5:7b"

    with _CAPABILITY_LOCK:
        if model in _MODEL_CAPABILITY:
            return _MODEL_CAPABILITY[model]

    server_url = (ollama_url or get_llm_host()).rstrip("/")
    backend = get_llm_backend()

    if backend == "vllm":
        try:
            r = requests.get(f"{server_url}/v1/models", timeout=3)
            if r.status_code == 200:
                available = [m.get("id") for m in r.json().get("data", [])]
                if model in available:
                    with _CAPABILITY_LOCK:
                        _MODEL_CAPABILITY[model] = "vllm_chat"
                    return "vllm_chat"
        except Exception:
            pass
        with _CAPABILITY_LOCK:
            _MODEL_CAPABILITY[model] = "none"
        return "none"

    try:
        # Step 1: Check for chat template via /api/show
        r = requests.post(
            f"{server_url}/api/show",
            json={"name": model},
            timeout=5,
        )
        if r.status_code == 404:
            with _CAPABILITY_LOCK:
                _MODEL_CAPABILITY[model] = "none"
            return "none"

        data = r.json() if r.status_code == 200 else {}
        has_template = bool(data.get("template", "").strip())

        if has_template:
            with _CAPABILITY_LOCK:
                _MODEL_CAPABILITY[model] = "chat"
            return "chat"

        # Step 2: Template absent — probe /api/generate (min-latency empty prompt)
        g = requests.post(
            f"{server_url}/api/generate",
            json={"model": model, "prompt": "", "stream": False},
            timeout=8,
        )
        cap = "generate" if g.status_code == 200 else "none"

    except Exception:
        cap = "none"

    with _CAPABILITY_LOCK:
        _MODEL_CAPABILITY[model] = cap
    return cap


def assert_model_ready(
    model: str,
    ollama_url: Optional[str] = None,
) -> str:
    """
    Verify the model is usable. Returns the capability ("vllm_chat", "chat", or "generate").

    If the model is not available or supports neither API, raises RuntimeError
    with exact operational guidance.
    """
    backend = get_llm_backend()
    server_url = (ollama_url or get_llm_host()).rstrip("/")

    if backend == "vllm":
        try:
            res = requests.get(f"{server_url}/v1/models", timeout=3)
        except Exception as e:
            raise RuntimeError(
                f"\n[LLM STARTUP GATE] Could not connect to vLLM server at {server_url}: {e}\n"
                f"Ensure vLLM is running:\n"
                f"  python -m vllm.entrypoints.openai.api_server --model {model} --port 8000\n"
            )
        if not res.ok:
            raise RuntimeError(
                f"\n[LLM STARTUP GATE] vLLM server at {server_url} returned HTTP {res.status_code}: {res.text[:200]}\n"
            )
        available = [m.get("id") for m in res.json().get("data", [])]
        if model not in available:
            raise RuntimeError(
                f"\n[LLM STARTUP GATE] vLLM is running at {server_url} but does not serve model '{model}'.\n"
                f"Available models: {available}\n"
                f"Either update AION_MODEL or restart vLLM with the correct --model flag.\n"
            )
        print(f"[LLM] Model {model!r}: capability=vllm_chat [OK]")
        with _CAPABILITY_LOCK:
            _MODEL_CAPABILITY[model] = "vllm_chat"
        return "vllm_chat"

    cap = probe_model_capability(model, server_url)
    if cap != "none":
        print(f"[LLM] Model {model!r}: capability={cap} [OK]")
        return cap

    # Derive the most likely instruct variant for the helpful error message
    instruct_variant = model if model.endswith("-instruct") else f"{model}-instruct"

    raise RuntimeError(
        f"\n"
        f"[LLM STARTUP GATE] Model {model!r} is NOT usable.\n"
        f"Ollama rejected both /api/chat and /api/generate for this model.\n"
        f"This usually means you pulled a base (non-instruct) model.\n"
        f"\n"
        f"To fix this, run ONE of the following and then restart AION:\n"
        f"\n"
        f"  ollama pull {instruct_variant}   ← try this first\n"
        f"  ollama pull qwen2.5:3b-instruct  ← recommended for laptop\n"
        f"  ollama pull qwen2.5:1.5b-instruct ← minimum (very low RAM)\n"
        f"\n"
        f"Then set AION_MODEL to the model you pulled, e.g.:\n"
        f"  $env:AION_MODEL = '{instruct_variant}'\n"
    )



def _get_concurrency() -> int:
    """
    Read concurrency from environment or active RuntimeProfile, with dynamic NVML VRAM throttling.
    If free VRAM drops below 10GB on GPU, throttles down to 4 concurrency.
    """
    target_c = 2
    env_c = os.getenv("AION_CONCURRENCY")
    if env_c and env_c.strip().isdigit():
        target_c = max(1, int(env_c.strip()))
    else:
        try:
            from runtime import get_active_profile
            target_c = get_active_profile().concurrency
        except Exception:
            target_c = 2

    # Dynamic NVML VRAM Throttling
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, _ = torch.cuda.mem_get_info()
            free_gb = free_bytes / (1024 ** 3)
            if free_gb < 10.0 and target_c > 4:
                return 4
    except Exception:
        pass

    return target_c


# Bounded concurrency semaphore — value set from runtime profile or env
_concurrency_val = _get_concurrency()
CONCURRENCY_SEMAPHORE = threading.Semaphore(_concurrency_val)
_sem_lock = threading.Lock()

def get_concurrency_semaphore() -> threading.Semaphore:
    """Returns dynamic concurrency semaphore reacting to environment/profile changes."""
    global CONCURRENCY_SEMAPHORE, _concurrency_val
    curr = _get_concurrency()
    if curr != _concurrency_val:
        with _sem_lock:
            if curr != _concurrency_val:
                _concurrency_val = curr
                CONCURRENCY_SEMAPHORE = threading.Semaphore(curr)
    return CONCURRENCY_SEMAPHORE


def get_best_llm():
    """
    Auto-selects the inference backend based on the active RuntimeProfile.

    - LAPTOP_FAST / LAPTOP_DEMO: Uses the benchmark-winning backend and model
      recorded in .aion_cache/runtime_profile.json.
    - PRODUCTION: Uses the standard RobustLLMCaller (Ollama / vLLM).
    - Legacy path: Falls back to OpenVINO if AION_USE_OPENVINO=1.

    Calls assert_model_ready() before constructing any caller.
    If the model is unusable, raises immediately with install instructions.
    """
    from pathlib import Path

    # -- Runtime profile integration ---------------------------------------
    try:
        from runtime import get_active_profile
        profile = get_active_profile()
        profile_name = profile.name
    except Exception:
        profile_name = os.environ.get("AION_PROFILE", "PRODUCTION")
        profile = None

    if profile is not None:
        pass  # profile.validate_environment() — validated at startup

    env_model = os.environ.get("AION_MODEL")
    if env_model and profile is not None and env_model not in profile.allowed_models:
        raise RuntimeError(
            f"[PROFILE INTEGRITY VIOLATION]\n"
            f"  Profile         : {profile_name}\n"
            f"  Allowed models  : {set(profile.allowed_models)}\n"
            f"  Requested model : {env_model}\n"
            f"  Action          : BLOCK — generation refused\n"
        )

    if profile_name in ("LAPTOP_FAST", "LAPTOP_DEMO") and profile is not None:
        model = env_model or profile.model_name
        if model == "AUTO":
            from core.config.production_model import get_production_model
            model = env_model or get_production_model()
        
        if model not in profile.allowed_models:
            raise RuntimeError(
                f"[PROFILE INTEGRITY VIOLATION]\n"
                f"  Profile         : {profile_name}\n"
                f"  Allowed models  : {set(profile.allowed_models)}\n"
                f"  Requested model : {model}\n"
                f"  Action          : BLOCK — generation refused\n"
            )
            
        backend = profile.backend
        timeout = int(profile.timeout_budget.per_slot)
        retries = profile.max_retries
        cap = assert_model_ready(model)   # blocks if not usable
        print(f"[LLM] Profile={profile_name}: backend={backend}, model={model}, "
              f"capability={cap}, timeout={timeout}s, retries={retries}")
        return RobustLLMCaller(
            primary_model=model,
            timeout_sec=timeout,
            max_retries=retries,
        )

    # -- Legacy OpenVINO path ------------------------------------------------
    ov_model = Path(__file__).parent.parent / "models" / "qwen2.5-7b-ov"

    if (os.environ.get("AION_USE_OPENVINO") == "1" or ov_model.exists()) and (ov_model / "openvino_model.xml").exists():
        try:
            from v0_1.openvino_llm import get_ov_llm
            print("[LLM] Using OpenVINO on Intel Arc iGPU (Laptop local accelerator)")
            return get_ov_llm(device="GPU")
        except Exception as e:
            print(f"[LLM] OpenVINO load warning: {e} — falling back to standard caller")

    model = env_model or get_production_model()
    if profile is not None and model not in profile.allowed_models:
        raise RuntimeError(
            f"[PROFILE INTEGRITY VIOLATION]\n"
            f"  Profile         : {profile_name}\n"
            f"  Allowed models  : {set(profile.allowed_models)}\n"
            f"  Requested model : {model}\n"
            f"  Action          : BLOCK — generation refused\n"
        )
    cap = assert_model_ready(model)       # blocks if not usable
    backend = get_llm_backend()
    print(f"[LLM] Using Standard RobustLLMCaller ({backend.upper()} / Production L40 GPU server): {model} ({cap})")
    return RobustLLMCaller(primary_model=model)


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
        primary_model:   Optional[str] = None,
        fallback_models: Optional[list[str]] = None,
        timeout_sec:     int = 180,
        max_retries:     int = 2,
        ollama_url:      Optional[str] = None,
        allow_fallback:  bool = False,
    ):
        self.backend        = get_llm_backend()
        self.primary_model  = primary_model or get_production_model()
        self.fallback_models = fallback_models or []
        self.allow_fallback  = allow_fallback
        self.timeout_sec    = timeout_sec
        self.max_retries    = max_retries
        self.ollama_url     = (ollama_url or get_llm_host()).rstrip("/")
        self._consecutive_failures = 0
        self.MAX_CONSECUTIVE_FAILURES = 3
        print(f"[LLM] RobustLLMCaller initialized — backend: {self.backend}, primary: {self.primary_model} on {self.ollama_url} "
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
        with get_concurrency_semaphore():
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
                server_name = "vLLM" if self.backend == "vllm" else "Ollama"
                hint = (
                    f"Ensure: python -m vllm.entrypoints.openai.api_server --model {get_production_model()} --port 8000"
                    if self.backend == "vllm"
                    else f"Ensure: ollama serve && ollama pull {get_production_model()}"
                )
                raise RuntimeError(
                    f"[LLM] ABORT: {self._consecutive_failures} consecutive LLM failures. "
                    f"Is {server_name} running? {hint}"
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
        # Route to the correct API endpoint based on cached capability
        capability = probe_model_capability(model, self.ollama_url)

        def _worker():
            try:
                try:
                    seed_val = int(os.environ.get("AION_SEED", "42"))
                except (ValueError, TypeError):
                    seed_val = 42

                if capability == "vllm_chat" or self.backend == "vllm":
                    vllm_model = model
                    if (":" in vllm_model or not vllm_model.startswith("Qwen/")) and os.environ.get("AION_MODEL"):
                        vllm_model = os.environ.get("AION_MODEL")
                    r = requests.post(
                        f"{self.ollama_url}/v1/chat/completions",
                        json={
                            "model":       vllm_model,
                            "messages":    [{"role": "user", "content": prompt}],
                            "max_tokens":  max_tokens,
                            "temperature": 0.1,
                            "seed":        seed_val,
                        },
                        timeout=timeout,
                    )
                    if r.status_code == 200:
                        data = r.json()
                        choices = data.get("choices", [])
                        content = ""
                        if choices:
                            content = choices[0].get("message", {}).get("content", "").strip()
                        result_queue.put(content if content else None)
                    else:
                        print(f"[LLM] vLLM HTTP {r.status_code}: {r.text[:200]}")
                        result_queue.put(None)

                elif capability == "chat":
                    r = requests.post(
                        f"{self.ollama_url}/api/chat",
                        json={
                            "model":    model,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream":   False,
                            "options": {
                                "num_predict":    min(max_tokens, 350),
                                "temperature":    0.1,
                                "seed":           seed_val,
                                "top_p":          0.9,
                                "top_k":          40,
                                "repeat_penalty": 1.1,
                                "num_thread":     14,
                                "num_batch":      512,
                            },
                        },
                        timeout=timeout,
                    )
                    if r.status_code == 200:
                        data    = r.json()
                        content = data.get("message", {}).get("content", "").strip()
                        if not content:
                            content = data.get("response", "").strip()
                        result_queue.put(content if content else None)
                    else:
                        print(f"[LLM] HTTP {r.status_code}: {r.text[:200]}")
                        result_queue.put(None)

                elif capability == "generate":
                    r = requests.post(
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model":  model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "num_predict":    min(max_tokens, 350),
                                "temperature":    0.1,
                                "top_p":          0.9,
                                "top_k":          40,
                                "repeat_penalty": 1.1,
                                "num_thread":     14,
                                "num_batch":      512,
                            },
                        },
                        timeout=timeout,
                    )
                    if r.status_code == 200:
                        content = r.json().get("response", "").strip()
                        result_queue.put(content if content else None)
                    else:
                        print(f"[LLM] HTTP {r.status_code}: {r.text[:200]}")
                        result_queue.put(None)

                else:
                    # capability == "none" — should have been blocked by assert_model_ready
                    print(f"[LLM] Model {model!r} has no usable capability. Run assert_model_ready() at startup.")
                    result_queue.put(None)

            except requests.Timeout as e:
                print(f"[LLM] {model} timed out after {timeout}s: {e}")
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

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        self.preferred_model = model or os.environ.get("AION_MODEL", get_production_model())
        self.caller = RobustLLMCaller(primary_model=self.preferred_model, ollama_url=host or get_llm_host())


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


