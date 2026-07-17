# aion_web/training/backend_registry.py
"""
Backend Registry — the single point of truth for which backend is
currently active.

Enforces mutual exclusivity: only one backend can be active at a time.
Switching while busy raises BackendBusyError.
"""

from __future__ import annotations

import threading
from typing import Optional

from aion_web.training.backends.base import (
    TrainingBackend, TrainingMode, BackendBusyError,
)
from aion_web.training.backends.demo_backend import DemoBackend
from aion_web.training.backends.local_backend import LocalBackend
from aion_web.training.backends.remote_backend import RemoteBackend
from aion_web.training.mode_config import ModeConfig


class BackendRegistry:
    """
    Thread-safe registry managing the active TrainingBackend.

    Usage:
        registry = BackendRegistry()
        registry.switch(TrainingMode.LOCAL, config)
        backend = registry.get()
        backend.analyse(files)
    """

    def __init__(self):
        self._active: Optional[TrainingBackend] = DemoBackend()
        self._lock = threading.RLock()

    def get(self) -> TrainingBackend:
        with self._lock:
            if self._active is None:
                raise RuntimeError("No backend is currently active.")
            return self._active

    def switch(self, mode: TrainingMode, config: ModeConfig) -> TrainingBackend:
        """
        Switch the active backend.

        Raises BackendBusyError if a job is currently running —
        the caller must call backend.cancel() first.
        """
        with self._lock:
            current = self._active

            # Guard: never switch while training is running
            if current is not None and current.is_busy:
                raise BackendBusyError(
                    f"Cannot switch to {mode.value} mode: "
                    f"{current.display_name} is currently running a job. "
                    f"Stop the current training before switching modes."
                )

            new_backend = self._build(mode, config)
            self._active = new_backend
            return new_backend

    def current_mode(self) -> Optional[TrainingMode]:
        with self._lock:
            return self._active.mode if self._active else None

    def _build(self, mode: TrainingMode, config: ModeConfig) -> TrainingBackend:
        if mode == TrainingMode.DEMO:
            return DemoBackend()
        elif mode == TrainingMode.LOCAL:
            return LocalBackend(
                model_name=config.local_model,
                ollama_url=config.ollama_url,
            )
        elif mode == TrainingMode.REMOTE:
            if not config.server_url:
                raise ValueError("Remote mode requires a server URL.")
            if not config.server_token:
                raise ValueError("Remote mode requires a server token.")
            return RemoteBackend(
                server_url=config.server_url,
                token=config.server_token,
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")
