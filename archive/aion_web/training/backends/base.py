# aion_web/training/backends/base.py
"""
TrainingBackend — the single interface the GUI calls for every training
operation. The GUI never knows which backend is active; it only calls
these methods.

Contract rules:
    1. Every method is synchronous from the caller's perspective.
       Long-running operations return a JobHandle immediately and
       stream progress through get_progress().
    2. Local and Remote backends NEVER use mock data.
       If an operation fails, they raise BackendError — they do not
       silently fall back to fake results.
    3. Demo backend NEVER touches real files, real models, or
       real servers. It simulates realistic timing but produces
       only fake structured output.
    4. Switching backends while a job is running raises
       BackendBusyError. The caller must stop the current job first.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional


class TrainingMode(str, Enum):
    DEMO = "demo"
    LOCAL = "local"
    REMOTE = "remote"


class ModeColor(str, Enum):
    DEMO = "orange"
    LOCAL = "green"
    REMOTE = "blue"

    @classmethod
    def for_mode(cls, mode: TrainingMode) -> str:
        return {
            TrainingMode.DEMO: cls.DEMO,
            TrainingMode.LOCAL: cls.LOCAL,
            TrainingMode.REMOTE: cls.REMOTE,
        }[mode]


class BackendError(Exception):
    """Raised when Local or Remote backends encounter a real failure.
    Never caught silently — always surfaced to the user."""


class BackendBusyError(BackendError):
    """Raised when a mode switch is attempted while a job is running."""


@dataclass
class JobHandle:
    job_id: str
    mode: TrainingMode
    subject_code: str = ""
    started_at: str = ""


@dataclass
class ProgressEvent:
    message: str
    fraction: float                  # 0.0 – 1.0
    stage: str = ""                  # current pipeline stage name
    metrics: Dict[str, Any] = field(default_factory=dict)
    is_terminal: bool = False        # True = job done (pass or fail)
    is_error: bool = False


@dataclass
class AnalysisOutput:
    session_id: str
    subject_code: str
    subject_name: str
    department: str
    semester: int
    books: int
    notes: int
    question_banks: int
    previous_papers: int
    module_summaries: List[Dict[str, Any]] = field(default_factory=list)
    ambiguities: List[Dict[str, Any]] = field(default_factory=list)
    train_enabled: bool = False
    mode: TrainingMode = TrainingMode.DEMO


@dataclass
class TrainingOutput:
    job_id: str
    model_version: str
    benchmark_scores: Dict[str, float] = field(default_factory=dict)
    can_promote: bool = False
    mode: TrainingMode = TrainingMode.DEMO


class TrainingBackend(abc.ABC):
    """Every backend implements this interface and nothing else."""

    mode: TrainingMode

    # -- Core operations ----------------------------------------------

    @abc.abstractmethod
    def analyse(
        self,
        file_paths: List[str],
        subject_code: str = "",
    ) -> AnalysisOutput:
        """
        Run document analysis. Returns immediately with structured output.
        Raises BackendError on failure (Local/Remote only).
        """

    @abc.abstractmethod
    def train(
        self,
        session_id: str,
        subject_code: str,
    ) -> JobHandle:
        """
        Start training. Returns a JobHandle immediately.
        Actual training is asynchronous; poll via get_progress().
        """

    @abc.abstractmethod
    def get_progress(self, job_id: str) -> Generator[ProgressEvent, None, None]:
        """
        Yield ProgressEvents until the job reaches a terminal state.
        Callers should handle StopIteration and is_terminal == True.
        """

    @abc.abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Cancel a running job. Returns True if successfully cancelled."""

    @abc.abstractmethod
    def resolve_ambiguity(
        self,
        session_id: str,
        ambiguity_id: str,
        action: str,
        value: Any,
    ) -> Dict[str, Any]:
        """Resolve one ambiguity from the analysis output."""

    @abc.abstractmethod
    def confirm_course(self, session_id: str) -> bool:
        """User has approved the course preview. Enables training."""

    @abc.abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Quick connectivity and readiness check.
        Returns {"healthy": bool, "details": str}
        """

    # -- Status -------------------------------------------------------

    @property
    @abc.abstractmethod
    def is_busy(self) -> bool:
        """True if a job is currently running on this backend."""

    @property
    def color(self) -> str:
        return ModeColor.for_mode(self.mode)

    @property
    def display_name(self) -> str:
        return {
            TrainingMode.DEMO: "Demo Mode",
            TrainingMode.LOCAL: "Local Training",
            TrainingMode.REMOTE: "Remote Server",
        }[self.mode]
