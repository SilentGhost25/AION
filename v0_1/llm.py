"""
AION Unified LLM Interface (Ollama Native).
Fixed: keep_alive at root level, better error reporting,
       correct model priority.
"""

from __future__ import annotations

import os
import json
import urllib.request
import urllib.error
from typing import Optional


class AIONLLM:
    def __init__(
        self,
        model: str = None,
        host:  str = "http://localhost:11434"
    ):
        self.preferred_model  = model or os.environ.get("AION_MODEL", "qwen2.5:3b")
        self.fallback_models  = [
            "qwen2.5:3b",
            "qwen2.5:1.5b",
            "llama3.2:3b",
            "mistral:latest"
        ]
        self.host = host.rstrip("/")
        print(f"[LLM] Initialized — preferred model: {self.preferred_model}")

    def generate(
        self,
        prompt:      str,
        system:      Optional[str] = None,
        temperature: float = 0.45,
        options:     Optional[dict] = None
    ) -> str:
        
        # Build options — NO keep_alive inside options
        opts = {"temperature": temperature}
        if options:
            clean_opts = {k: v for k, v in options.items() if k != "keep_alive"}
            opts.update(clean_opts)

        candidate_models = [self.preferred_model] + [
            m for m in self.fallback_models
            if m != self.preferred_model
        ]

        system_prompt = system or (
            "You are AION, an academic question generator for VTU engineering exams."
        )

        # ── Strategy 1: Python ollama package ──────────────────
        try:
            import ollama
            import threading

            for mdl in candidate_models:
                try:
                    messages = [
                        {"role": "system",  "content": system_prompt},
                        {"role": "user",    "content": prompt}
                    ]

                    result_holder = [None]
                    error_holder  = [None]

                    def _call():
                        try:
                            res = ollama.chat(
                                model      = mdl,
                                messages   = messages,
                                options    = opts,
                                keep_alive = -1
                            )
                            result_holder[0] = (
                                res.get("message", {})
                                   .get("content", "")
                                   .strip()
                            )
                        except Exception as err:
                            error_holder[0] = err

                    t = threading.Thread(target=_call, daemon=True)
                    t.start()
                    t.join(timeout=90)

                    if t.is_alive():
                        print(f"[LLM] [WARNING] Timeout for {mdl} — skipping to next backend")
                        continue

                    if error_holder[0]:
                        raise error_holder[0]

                    content = result_holder[0] or ""
                    if content:
                        print(f"[LLM] Generated via ollama package ({mdl}): {len(content)} chars")
                        return content
                except Exception as e:
                    print(f"[LLM] ollama.chat failed for {mdl}: {e}")
                    continue

        except ImportError:
            print("[LLM] ollama package not installed, using HTTP fallback")
        except Exception as e:
            print(f"[LLM] ollama package note: {e}")

        # ── Strategy 2: Direct HTTP to Ollama API ──────────────
        for mdl in candidate_models:
            try:
                payload = {
                    "model":      mdl,
                    "messages":   [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt}
                    ],
                    "stream":     False,
                    "keep_alive": -1,    # Root level - correct position
                    "options":    opts   # No keep_alive inside options
                }

                req = urllib.request.Request(
                    f"{self.host}/api/chat",
                    data    = json.dumps(payload).encode("utf-8"),
                    headers = {"Content-Type": "application/json"}
                )

                with urllib.request.urlopen(req, timeout=120) as res:
                    if res.status == 200:
                        data    = json.loads(res.read().decode("utf-8"))
                        content = (
                            data.get("message", {})
                                .get("content", "")
                                .strip()
                        )
                        if content:
                            print(f"[LLM] Generated via HTTP ({mdl}): {len(content)} chars")
                            return content

            except Exception as e:
                print(f"[LLM] HTTP error for {mdl}: {e}")

        print("[LLM] ALL backends failed — returning empty string")
        return ""


# ── Singleton ───────────────────────────────────────────────
_default_llm: Optional[AIONLLM] = None


def get_llm(model: Optional[str] = None) -> AIONLLM:
    global _default_llm
    if _default_llm is None or (
        model and _default_llm.preferred_model != model
    ):
        _default_llm = AIONLLM(model=model)
    return _default_llm
