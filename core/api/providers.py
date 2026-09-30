"""
Declarative provider definitions for the AION API layer.

A provider is a named OpenAI-compatible endpoint with:
  - rate limits (RPM, RPD)
  - capability lists (which models support vision vs. text)
  - a priority for tie-breaking
  - an env var name that holds the API key

The router never calls the network. It only selects a provider and
returns it; the caller is responsible for the actual HTTP request.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class APIProvider:
    name: str
    base_url: str
    api_key_env: str
    rpm_limit: int
    rpd_limit: Optional[int]            # None = no published daily limit
    text_models: List[str]
    vision_models: List[str] = field(default_factory=list)
    priority: int = 100                 # lower = preferred in tie-breaks

    @property
    def supports_vision(self) -> bool:
        return len(self.vision_models) > 0

    def best_text_model(self) -> str:
        return self.text_models[0] if self.text_models else ""

    def best_vision_model(self) -> str:
        return self.vision_models[0] if self.vision_models else ""


# -----------------------------------------------------------------------------
# Default registry — matches the plan discussed earlier.
# Priority ordering (lower wins when both are eligible):
#   10  Groq            — fastest, good text, vision is limited
#   20  NVIDIA NIM      — best vision, competitive text
#   30  Google AI       — strongest models, lowest free-tier quota
#   40  OpenRouter      — fallback aggregator
#   50  Ollama (local)  — offline safety net, no rate limit
# -----------------------------------------------------------------------------

PROVIDERS: List[APIProvider] = [
    APIProvider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        rpm_limit=30,
        rpd_limit=14_400,
        text_models=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ],
        vision_models=[
            "llama-3.2-11b-vision-preview",
        ],
        priority=10,
    ),
    APIProvider(
        name="nvidia_nim",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        rpm_limit=40,
        rpd_limit=None,
        text_models=[
            "meta/llama-3.3-70b-instruct",
            "deepseek-ai/deepseek-v3.2",
        ],
        vision_models=[
            "meta/llama-3.2-11b-vision-instruct",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        ],
        priority=20,
    ),
    APIProvider(
        name="google_ai",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GOOGLE_AI_API_KEY",
        rpm_limit=15,
        rpd_limit=500,
        text_models=[
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ],
        vision_models=[
            "gemini-2.5-flash",
        ],
        priority=30,
    ),
    APIProvider(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        rpm_limit=20,
        rpd_limit=50,
        text_models=[
            "meta-llama/llama-3.3-70b-instruct:free",
        ],
        vision_models=[],
        priority=40,
    ),
    APIProvider(
        name="ollama_local",
        base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_API_KEY",       # may be empty; Ollama ignores it
        rpm_limit=10_000,                    # effectively unlimited
        rpd_limit=None,
        text_models=["qwen2.5:14b-instruct"],
        vision_models=["qwen2.5-vl:7b"],
        priority=50,
    ),
]
