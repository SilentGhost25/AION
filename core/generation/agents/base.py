"""
Base agent contract.

Every agent implements `run(context) -> AgentResult`. The context is
a mutable bag that flows through the pipeline; the result is a
structured outcome with success flag, payload, and failure codes.

Design notes:
    - No agent raises on expected failure. Failures are AgentResults
      with `success=False` and a `failure_code`. Raising is reserved
      for programming errors (missing dependency, invalid config).
    - Telemetry is a free-form dict on both context and result. The
      Checking Agent aggregates it into the QA report.
    - Agents are stateless beyond configuration. All per-request state
      lives in the context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentResult:
    success: bool
    payload: Any = None
    failure_code: Optional[str] = None
    failure_detail: Optional[str] = None
    telemetry: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.success and not self.failure_code:
            raise ValueError("failed AgentResult must have a failure_code")


@dataclass
class AgentContext:
    """Mutable state passed through the pipeline. Not shared across requests."""
    # Request-level inputs
    request: dict = field(default_factory=dict)

    # Contracts and evidence
    paper_spec: Any = None
    artifact: Any = None
    knowledge_graph: Any = None
    marks_split: list = field(default_factory=list)

    # Stage outputs
    plans: list = field(default_factory=list)
    questions: list = field(default_factory=list)
    audit_report: Any = None
    qa_report: Any = None                       # NEW — set by Checking Agent

    # Telemetry accumulates across agents
    telemetry: dict = field(default_factory=dict)

    # Failure tracking (set by Checking Agent)
    unresolved_slots: list = field(default_factory=list)


class Agent(ABC):
    """Base class for all agents."""

    name: str = "agent"
    version: str = "v3.0"

    @abstractmethod
    def run(self, context: AgentContext) -> AgentResult:
        ...

    def log(self, event: str, **kwargs: Any) -> None:
        """Structured log line. Agents call this for observability."""
        parts = [f"[AGENT:{self.name}]", event]
        for k, v in kwargs.items():
            parts.append(f"{k}={v}")
        print(" ".join(parts), flush=True)
