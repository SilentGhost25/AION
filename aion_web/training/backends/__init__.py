# aion_web/training/backends/__init__.py
from __future__ import annotations

from aion_web.training.backends.base import (
    TrainingBackend, TrainingMode, ModeColor, BackendError,
    BackendBusyError, JobHandle, ProgressEvent, AnalysisOutput, TrainingOutput,
)
from aion_web.training.backends.demo_backend import DemoBackend
from aion_web.training.backends.local_backend import LocalBackend
from aion_web.training.backends.remote_backend import RemoteBackend

__all__ = [
    "TrainingBackend",
    "TrainingMode",
    "ModeColor",
    "BackendError",
    "BackendBusyError",
    "JobHandle",
    "ProgressEvent",
    "AnalysisOutput",
    "TrainingOutput",
    "DemoBackend",
    "LocalBackend",
    "RemoteBackend",
]
