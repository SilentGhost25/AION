"""
AION Single-Request LLM Wrapper
================================
Forces Ollama into single-threaded mode using a Python threading.Lock.
Prevents queue buildup and 503 Server Busy errors entirely.
"""

import requests
import json
import time
import threading
from typing import Optional


class SingleRequestLLM:
    """
    Only one request at a time.
    All other requests wait in Python queue, not Ollama queue.
    """

    def __init__(self, model: str = "qwen2.5:1.5b", base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._lock = threading.Lock()
        self._request_count = 0

    def generate(self, prompt: str, max_tokens: int = 128) -> Optional[str]:
        """
        Acquire lock before calling Ollama.
        Ensures Ollama queue never has > 1 request.
        """
        self._request_count += 1
        req_id = self._request_count

        print(f"[LLM #{req_id}] Waiting for lock...", flush=True)

        with self._lock:
            print(f"[LLM #{req_id}] Lock acquired, calling Ollama...", flush=True)

            result = self._generate_internal(prompt, max_tokens, req_id)
            self._force_unload()
            return result

    def _generate_internal(self, prompt: str, max_tokens: int, req_id: int) -> Optional[str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.2,
            },
            "keep_alive": 0,  # Unload immediately
        }

        try:
            start = time.time()
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=30
            )

            if response.status_code == 503:
                print(f"[LLM #{req_id}] 503 Server Busy — Ollama queue corrupted", flush=True)
                print(
                    f"[LLM #{req_id}] ACTION REQUIRED:\n"
                    f"  1. Stop script (Ctrl+C)\n"
                    f"  2. Run: taskkill /F /IM ollama.exe\n"
                    f"  3. Delete: %LOCALAPPDATA%\\Ollama\\*.db\n"
                    f"  4. Run: ollama serve",
                    flush=True
                )
                return None

            if response.status_code != 200:
                print(f"[LLM #{req_id}] HTTP {response.status_code}", flush=True)
                return None

            tokens = []
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("response", "")
                    tokens.append(token)
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

            result = "".join(tokens).strip()
            elapsed = time.time() - start

            print(f"[LLM #{req_id}] ✓ {len(tokens)} tokens in {elapsed:.1f}s", flush=True)
            return result

        except requests.Timeout:
            print(f"[LLM #{req_id}] Timeout", flush=True)
            return None

        except Exception as e:
            print(f"[LLM #{req_id}] Error: {e}", flush=True)
            return None

    def _force_unload(self):
        """Send explicit unload command."""
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": 0,
                },
                timeout=5
            )
            time.sleep(1)
        except Exception:
            pass


llm = SingleRequestLLM()
