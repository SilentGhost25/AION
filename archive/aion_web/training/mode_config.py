# aion_web/training/mode_config.py
"""
ModeConfig — validated configuration for all three training modes.
Stored in the Flask app's config / settings DB.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from aion_web.training.backends.base import TrainingMode


@dataclass
class ModeConfig:
    active_mode: str = TrainingMode.DEMO.value  # stored as string for JSON compat

    # Remote mode
    server_url: str = ""
    server_token: str = ""
    auto_connect: bool = True

    # Local mode
    local_model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434"

    @property
    def mode(self) -> TrainingMode:
        return TrainingMode(self.active_mode)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "ModeConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: str):
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str) -> "ModeConfig":
        if not Path(path).exists():
            return cls()
        return cls.from_dict(json.loads(Path(path).read_text()))
