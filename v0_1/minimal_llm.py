"""
AION Minimal LLM Caller
========================
Bare-bones LLM caller with zero extras.
Single model, single request, token streaming, 5s token hang detection, immediate unload.
"""

import requests
import json
import time
from typing import Optional


class MinimalLLM:
    """
    Rock-solid LLM caller.
    No retries. No fallbacks. No complexity.
    If it fails, it fails FAST and LOUD.
    """

    def __init__(self, model: str = "qwen2.5:1.5b", base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128
    ) -> Optional[str]:
        """
        ONE attempt. ONE model. Returns fast.
        """
        if not self._health_check():
            raise RuntimeError(
                "Ollama not responding. "
                "Run: ollama serve"
            )

        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": True,  # Stream for early hang detection
            "options": {
                "num_predict":  max_tokens,
                "temperature":  0.2,  # Low = predictable
                "top_p":        0.9,
                "top_k":        40,
            },
            "keep_alive": 0,  # Unload immediately after
        }

        print(f"[LLM] Calling {self.model}...", flush=True)
        start_time = time.time()

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=25  # Hard limit
            )

            if response.status_code != 200:
                print(f"[LLM] HTTP {response.status_code}: {response.text[:100]}", flush=True)
                return None

            tokens = []
            last_token_time = time.time()
            TOKEN_TIMEOUT = 5  # 5s between tokens max

            for line in response.iter_lines():
                if not line:
                    continue

                now = time.time()
                if now - last_token_time > TOKEN_TIMEOUT:
                    print(f"[LLM] Hung: no token for {TOKEN_TIMEOUT}s", flush=True)
                    response.close()
                    return None

                last_token_time = now

                try:
                    data = json.loads(line)
                    token = data.get("response", "")
                    tokens.append(token)

                    if data.get("done", False):
                        break

                except json.JSONDecodeError:
                    continue

            result = "".join(tokens).strip()
            elapsed = time.time() - start_time

            if result:
                print(f"[LLM] ✓ {len(tokens)} tokens in {elapsed:.1f}s", flush=True)
                return result
            else:
                print("[LLM] Empty response", flush=True)
                return None

        except requests.Timeout:
            print("[LLM] Timeout after 25s", flush=True)
            return None

        except requests.ConnectionError:
            print("[LLM] Connection failed — is Ollama running?", flush=True)
            return None

        except Exception as e:
            print(f"[LLM] Error: {type(e).__name__}: {e}", flush=True)
            return None

    def _health_check(self) -> bool:
        """Quick ping to Ollama."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def verify_model_loaded(self) -> bool:
        """Check if target model exists."""
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code != 200:
                return False

            models = r.json().get("models", [])
            model_names = [m.get("name", "") for m in models]

            exists = any(self.model in name for name in model_names)

            if not exists:
                print(
                    f"[LLM] Model {self.model} not found.\n"
                    f"      Available: {model_names}\n"
                    f"      Run: ollama pull {self.model}",
                    flush=True
                )

            return exists

        except Exception as e:
            print(f"[LLM] Model check failed: {e}", flush=True)
            return False


# Singleton instance
llm = MinimalLLM()
