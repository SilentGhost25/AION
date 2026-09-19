# core/generation/robust_llm_caller.py

import os, time, json, logging, re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

LOG = logging.getLogger("aion.llm")


def _default_seed() -> int:
    try:
        return int(os.environ.get("AION_SEED", "42"))
    except (ValueError, TypeError):
        return 42


@dataclass
class LLMRequest:
    model         : str
    prompt        : str
    schema        : Optional[dict] = None
    temperature   : float = 0.1
    seed          : int = field(default_factory=_default_seed)
    timeout_sec   : int = 45
    max_tokens    : int = 4096


@dataclass
class LLMResponse:
    success     : bool
    text        : Optional[str]
    parsed      : Optional[dict]
    elapsed_sec : float
    error       : Optional[str] = None
    timed_out   : bool = False


class RobustLLMCaller:
    """
    ONE inference request per call.
    Does NOT own retry logic — SlotOrchestrator owns all retries.
    Supports both Ollama and vLLM backends.
    """

    def __init__(self, backend: Optional[str] = None, host: Optional[str] = None):
        self.backend = (backend or os.environ.get("AION_BACKEND", "ollama")).lower().strip()
        if host:
            self.host = host.rstrip("/")
        else:
            if self.backend == "vllm":
                self.host = (os.environ.get("AION_LLM_HOST") or os.environ.get("VLLM_URL") or "http://localhost:8000").rstrip("/")
            else:
                self.host = (os.environ.get("OLLAMA_URL") or os.environ.get("OLLAMA_HOST") or os.environ.get("AION_LLM_HOST") or "http://localhost:11434").rstrip("/")

    def check_health(self) -> bool:
        """
        Pre-flight readiness probe verifying that the server is reachable AND
        is serving the expected model.
        """
        import requests as req
        from core.config.production_model import get_production_model
        model = os.environ.get("AION_MODEL") or get_production_model()

        if self.backend == "vllm":
            try:
                res = req.get(f"{self.host}/v1/models", timeout=3)
                if not res.ok:
                    raise RuntimeError(f"vLLM server at {self.host} returned HTTP {res.status_code}")
                available = [m.get("id") for m in res.json().get("data", [])]
                if model not in available:
                    raise RuntimeError(
                        f"[LLM STARTUP GATE] vLLM is running but does not serve model '{model}'.\n"
                        f"Available models: {available}\n"
                        f"Either update AION_MODEL or restart vLLM with the correct --model flag."
                    )
                return True
            except Exception as e:
                LOG.error(f"[LLM HEALTH] vLLM health check failed: {e}")
                raise
        else:
            try:
                res = req.get(f"{self.host}/api/tags", timeout=3)
                if not res.ok:
                    raise RuntimeError(f"Ollama server at {self.host} returned HTTP {res.status_code}")
                available = [m.get("name") for m in res.json().get("models", [])]
                if not any(model == m or model in m for m in available if m):
                    LOG.warning(f"[LLM HEALTH] Model '{model}' not found in Ollama tags {available}")
                return True
            except Exception as e:
                LOG.error(f"[LLM HEALTH] Ollama health check failed: {e}")
                raise

    def call(self, request: LLMRequest) -> LLMResponse:
        start = time.monotonic()
        try:
            if self.backend == "ollama":
                return self._call_ollama(request, start)
            elif self.backend == "vllm":
                return self._call_vllm(request, start)
            raise ValueError(f"Unknown backend: {self.backend}")
        except TimeoutError:
            elapsed = time.monotonic() - start
            LOG.warning(f"[LLM] Timeout after {elapsed:.1f}s for {request.model}")
            return LLMResponse(
                success=False, text=None, parsed=None,
                elapsed_sec=elapsed, timed_out=True,
                error="LLM_TIMEOUT"
            )
        except Exception as e:
            elapsed = time.monotonic() - start
            LOG.error(f"[LLM] Error: {e}")
            return LLMResponse(
                success=False, text=None, parsed=None,
                elapsed_sec=elapsed, error=str(e)
            )

    def _call_ollama(self, request: LLMRequest, start: float) -> LLMResponse:
        import requests as req
        payload: Dict[str, Any] = {
            "model"  : request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream"  : False,
            "options" : {
                "temperature": request.temperature,
                "seed"       : request.seed,
                "num_ctx"    : 8192,
            }
        }
        if request.schema:
            payload["format"] = request.schema

        try:
            response = req.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=request.timeout_sec,
            )
        except req.exceptions.Timeout:
            raise TimeoutError(f"Request to Ollama timed out after {request.timeout_sec}s")

        elapsed = time.monotonic() - start

        if not response.ok:
            return LLMResponse(
                success=False, text=None, parsed=None,
                elapsed_sec=elapsed, error=f"HTTP {response.status_code}"
            )

        resp_data = response.json()
        raw_text = resp_data.get("message", {}).get("content", "") or resp_data.get("response", "")

        # Truncation check
        max_tok = getattr(request, "max_tokens", 4096)
        if not raw_text.strip().endswith("}") and len(raw_text.split()) >= max_tok * 0.9:
            LOG.warning(f"[LLM] Generation truncated at max_tokens={max_tok} ({len(raw_text.split())} words generated).")
            return LLMResponse(
                success=False, text=raw_text, parsed=None,
                elapsed_sec=elapsed, error="TRUNCATED_AT_MAX_TOKENS"
            )

        # Extract JSON from response
        try:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            parsed = json.loads(match.group()) if match else json.loads(raw_text)
            return LLMResponse(
                success=True, text=raw_text, parsed=parsed, elapsed_sec=elapsed
            )
        except json.JSONDecodeError as e:
            return LLMResponse(
                success=False, text=raw_text, parsed=None,
                elapsed_sec=elapsed, error=f"JSON_PARSE_ERROR: {e}"
            )

    def _call_vllm(self, request: LLMRequest, start: float) -> LLMResponse:
        import requests as req
        max_tok = getattr(request, "max_tokens", 4096)
        payload: Dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": max_tok,
            "temperature": request.temperature,
            "seed": request.seed,
        }
        if request.schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "schema": request.schema,
                    "strict": True,
                }
            }
        else:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = req.post(
                f"{self.host}/v1/chat/completions",
                json=payload,
                timeout=request.timeout_sec,
            )
        except req.exceptions.Timeout:
            raise TimeoutError(f"Request to vLLM timed out after {request.timeout_sec}s")

        elapsed = time.monotonic() - start

        if not response.ok:
            # Fallback for vLLM versions that only support json_object
            if response.status_code == 400 and request.schema:
                LOG.warning(
                    f"[LLM] vLLM returned HTTP 400 on 'json_schema' response_format. "
                    f"Falling back to 'json_object'. Server details: {response.text[:200]}. "
                    f"Note: Verify vLLM version (>=0.5.4 recommended for native structured outputs)."
                )
                try:
                    payload["response_format"] = {"type": "json_object"}
                    response = req.post(
                        f"{self.host}/v1/chat/completions",
                        json=payload,
                        timeout=request.timeout_sec,
                    )
                    elapsed = time.monotonic() - start
                except Exception:
                    pass

            if not response.ok:
                return LLMResponse(
                    success=False, text=None, parsed=None,
                    elapsed_sec=elapsed, error=f"HTTP {response.status_code}: {response.text[:200]}"
                )

        resp_data = response.json()
        choices = resp_data.get("choices", [])
        raw_text = ""
        if choices:
            raw_text = choices[0].get("message", {}).get("content", "") or ""

        # Truncation check
        if not raw_text.strip().endswith("}") and len(raw_text.split()) >= max_tok * 0.9:
            LOG.warning(f"[LLM] vLLM generation truncated at max_tokens={max_tok} ({len(raw_text.split())} words generated).")
            return LLMResponse(
                success=False, text=raw_text, parsed=None,
                elapsed_sec=elapsed, error="TRUNCATED_AT_MAX_TOKENS"
            )

        # Extract JSON from response
        try:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            parsed = json.loads(match.group()) if match else json.loads(raw_text)
            return LLMResponse(
                success=True, text=raw_text, parsed=parsed, elapsed_sec=elapsed
            )
        except json.JSONDecodeError as e:
            return LLMResponse(
                success=False, text=raw_text, parsed=None,
                elapsed_sec=elapsed, error=f"JSON_PARSE_ERROR: {e}"
            )
