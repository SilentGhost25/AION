"""
VLM caller for the Writing Agent's figure-aware slots.

Single-endpoint, OpenAI-compatible. Sends the prompt plus an inline
image (base64 data URL). Reuses the same request/response contract as
HTTPAPICaller but with the multimodal message format.

Public surface:
    VLMCaller(base_url, model, ...)
    .call(prompt, image_path, schema=None, seed=None) -> parsed content
    .last_usage -> Optional[dict]

Raises:
    APIUnavailable  — on network failure, non-200, or exhausted retries
"""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import requests
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "core.api.vlm_caller requires requests. "
        "Install with: pip install requests"
    ) from e

# Pillow is optional — if not installed, images are sent as-is.
try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------


class APIUnavailable(RuntimeError):
    """Raised when the VLM endpoint cannot serve the request."""
    pass


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4096
RETRY_DELAY_SECONDS = 0.5

# Image resize limits — long edge capped to this many pixels.
DEFAULT_MAX_IMAGE_DIMENSION = 1024

# If Pillow is not available, skip images larger than this many bytes.
DEFAULT_MAX_IMAGE_BYTES = 3 * 1024 * 1024  # 3 MB


# -----------------------------------------------------------------------------
# Caller
# -----------------------------------------------------------------------------


class VLMCaller:
    """
    Vision-language model caller for figure-aware question generation.

    Parameters
    ----------
    base_url : str
        OpenAI-compatible VLM endpoint (e.g. http://localhost:8001/v1).
    model : str
        Model ID served by the endpoint.
    api_key : str | None
        Optional bearer token. Most local VLMs ignore this.
    timeout_seconds : float
        Per-request timeout. VLM inference is slower than text.
    max_attempts : int
        Retries on transient failure. Default 3.
    temperature : float
        Sampling temperature. Default 0.1.
    max_tokens : int
        Max completion tokens. Default 4096.
    seed : int | None
        Reproducibility seed. Default 42.
    max_image_dimension : int
        Long-edge cap in pixels. Default 1024.
    session : requests.Session | None
        Optional session for pooling and test injection.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        seed: Optional[int] = 42,
        max_image_dimension: int = DEFAULT_MAX_IMAGE_DIMENSION,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not base_url or not base_url.strip():
            raise ValueError("base_url is required")
        if not model or not model.strip():
            raise ValueError("model is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if not (0.0 <= temperature <= 2.0):
            raise ValueError("temperature must be in [0, 2]")
        if max_image_dimension < 64:
            raise ValueError("max_image_dimension must be >= 64")

        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = (api_key or "").strip()
        self._timeout = timeout_seconds
        self._max_attempts = max_attempts
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._seed = seed
        self._max_image_dimension = max_image_dimension
        self._session = session or requests.Session()

        self._last_usage: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ API

    @property
    def last_usage(self) -> Optional[Dict[str, Any]]:
        return self._last_usage

    def call(
        self,
        prompt: str,
        image_path: str,
        schema: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> Any:
        """
        Send prompt + image to the VLM.

        Returns parsed JSON if the content is JSON, else the raw string.
        Raises APIUnavailable if the endpoint fails after max_attempts.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be non-empty")
        if not image_path:
            raise ValueError("image_path is required")

        image_data_url = self._encode_image(image_path)
        last_error: Optional[str] = None

        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._post(
                    prompt=prompt,
                    image_data_url=image_data_url,
                    schema=schema,
                    seed=seed,
                )
            except requests.Timeout:
                last_error = f"timeout after {self._timeout}s"
                self._log(attempt, "TIMEOUT")
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            except requests.ConnectionError as e:
                last_error = f"connection error: {e}"
                self._log(attempt, "CONN_ERR")
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            except requests.RequestException as e:
                last_error = f"request error: {e}"
                self._log(attempt, "REQ_ERR")
                time.sleep(RETRY_DELAY_SECONDS)
                continue

            if response.status_code == 200:
                try:
                    content, usage = self._parse_response(response)
                except requests.RequestException as e:
                    last_error = str(e)
                    self._log(attempt, "MALFORMED_200")
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
                self._last_usage = usage
                self._log(attempt, "OK")
                return self._maybe_parse_json(content)

            if response.status_code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {response.status_code} ({response.text[:200]})"
                self._log(attempt, f"HTTP_{response.status_code}")
                time.sleep(RETRY_DELAY_SECONDS)
                continue

            # Config error — no point retrying
            last_error = f"HTTP {response.status_code} (config) {response.text[:200]}"
            self._log(attempt, f"CONFIG_{response.status_code}")
            break

        raise APIUnavailable(f"VLM endpoint failed: {last_error}")

    # ------------------------------------------------------------- Image

    def _encode_image(self, image_path: str) -> str:
        """
        Load the image, resize if oversized, and return a base64 data URL.

        Uses Pillow if available; otherwise sends raw bytes and hopes
        the VLM accepts them. If the file is too large without Pillow,
        raises ValueError rather than burning a request that will 413.
        """
        p = Path(image_path)
        if not p.exists():
            raise ValueError(f"image_path does not exist: {image_path}")
        if not p.is_file():
            raise ValueError(f"image_path is not a file: {image_path}")

        raw = p.read_bytes()
        mime = mimetypes.guess_type(p.name)[0] or "image/png"

        if _PIL_AVAILABLE:
            raw, mime = self._resize_if_needed(raw, mime)
        else:
            if len(raw) > DEFAULT_MAX_IMAGE_BYTES:
                raise ValueError(
                    f"image too large ({len(raw)} bytes) and Pillow is not "
                    f"installed. Install Pillow or supply a smaller image."
                )

        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def _resize_if_needed(self, raw: bytes, mime: str) -> tuple:
        """Resize the image if its long edge exceeds the cap."""
        try:
            img = Image.open(io.BytesIO(raw))
        except Exception:
            # Unrecognized format — send as-is
            return raw, mime

        w, h = img.size
        long_edge = max(w, h)
        if long_edge <= self._max_image_dimension:
            return raw, mime

        scale = self._max_image_dimension / long_edge
        new_size = (int(w * scale), int(h * scale))
        img = img.convert("RGB") if img.mode not in ("RGB", "RGBA") else img
        img = img.resize(new_size, Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue(), "image/png"

    # -------------------------------------------------------------- HTTP

    def _post(
        self,
        prompt: str,
        image_data_url: str,
        schema: Optional[dict],
        seed: Optional[int],
    ) -> requests.Response:
        url = self._base_url + "/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        body: Dict[str, Any] = {
            "model": self._model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        effective_seed = seed if seed is not None else self._seed
        if effective_seed is not None:
            body["seed"] = effective_seed

        if schema is not None:
            body["response_format"] = {"type": "json_object"}

        return self._session.post(
            url,
            headers=headers,
            json=body,
            timeout=self._timeout,
        )

    def _parse_response(self, response: requests.Response) -> tuple:
        try:
            body = response.json()
        except (ValueError, json.JSONDecodeError) as e:
            raise requests.RequestException(
                f"VLM returned non-JSON body: {e}"
            )

        choices = body.get("choices") or []
        if not choices:
            raise requests.RequestException(f"VLM returned no choices: {body}")

        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            raise requests.RequestException(
                f"VLM returned no content: {choices[0]}"
            )

        usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
        return str(content), usage

    def _maybe_parse_json(self, content: str) -> Any:
        import re
        text = content.strip()
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        if not text.startswith("{") and not text.startswith("["):
            m = re.search(r"[\{\[].*[\}\]]", text, re.DOTALL)
            if m:
                text = m.group(0)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return content

    def _log(self, attempt: int, status: str) -> None:
        print(
            f"[VLM-API] attempt={attempt} model={self._model} status={status}",
            flush=True,
        )
