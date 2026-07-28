"""
AION Unified LLM Interface (Ollama Native).
Provides instant, local, zero-download inference using Ollama.
Auto-falls back across installed models (qwen2.5:3b, llama3.2:3b, qwen2.5:1.5b, mistral:latest).
"""

from __future__ import annotations

import os
import json
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Any


class AIONLLM:
    """
    Unified LLM wrapper for AION pipeline.
    Primary backend: Ollama local models with auto model fallbacks.
    """

    def __init__(self, model: str = None, host: str = "http://localhost:11434"):
        self.preferred_model = model or os.environ.get("AION_MODEL", "llama3.2:3b")
        self.fallback_models = ["llama3.2:3b", "qwen2.5:3b", "qwen2.5:1.5b", "mistral:latest"]
        self.host = host.rstrip("/")

    def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.45, options: Optional[dict] = None) -> str:
        """
        Generate text from local Ollama model with model fallbacks and optional stop sequences / limits.
        """
        opts = {"temperature": temperature}
        if options:
            opts.update(options)

        candidate_models = [self.preferred_model] + [m for m in self.fallback_models if m != self.preferred_model]

        # Strategy 1: Python ollama package
        try:
            import ollama
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            for mdl in candidate_models:
                try:
                    response = ollama.chat(
                        model=mdl,
                        messages=messages,
                        options=opts
                    )
                    content = response.get("message", {}).get("content", "").strip()
                    if content:
                        return content
                except Exception:
                    try:
                        response = ollama.generate(
                            model=mdl,
                            prompt=f"{system}\n\n{prompt}" if system else prompt,
                            options=opts
                        )
                        content = response.get("response", "").strip()
                        if content:
                            return content
                    except Exception:
                        continue
        except ImportError:
            pass
        except Exception as e:
            print(f"[LLM] Ollama package note: {e}")

        # Strategy 2: Direct HTTP request to Ollama API
        for mdl in candidate_models:
            try:
                url = f"{self.host}/api/chat"
                payload = {
                    "model": mdl,
                    "messages": [
                        {"role": "system", "content": system or "You are AION, an academic question generator for VTU engineering exams."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": opts
                }

                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )

                with urllib.request.urlopen(req, timeout=120) as res:
                    if res.status == 200:
                        data = json.loads(res.read().decode("utf-8"))
                        content = data.get("message", {}).get("content", "").strip()
                        if content:
                            return content
            except Exception:
                continue

        print("[LLM] ⚠ All local Ollama model backends failed. Using template fallback.")
        return ""


# Global singleton instance
_default_llm: Optional[AIONLLM] = None


def get_llm(model: Optional[str] = None) -> AIONLLM:
    global _default_llm
    if _default_llm is None or (model and _default_llm.preferred_model != model):
        _default_llm = AIONLLM(model=model)
    return _default_llm
