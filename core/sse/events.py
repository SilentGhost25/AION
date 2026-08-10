"""
AION Core SSE Protocol — Terminal Event Contracts
===================================================
Defines SSEEventType, SSEEvent, and standard factory functions
for terminal success/failure events as specified in Part I.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class SSEEventType(str, Enum):
    # ── LIFECYCLE EVENTS ──────────────────────────────────────────────────────
    CONNECTED          = "connected"          # SSE channel open
    PIPELINE_STARTED   = "pipeline_started"   # first deterministic step complete
    STAGE_STARTED      = "stage_started"      # named stage beginning
    STAGE_COMPLETE     = "stage_complete"     # named stage succeeded
    PROGRESS           = "progress"           # incremental update

    # ── CONTENT EVENTS ────────────────────────────────────────────────────────
    QUESTION_READY     = "question_ready"     # one APPROVED question
    PAPER_READY        = "paper_ready"        # all questions APPROVED

    # ── TERMINAL EVENTS (exactly one is always emitted) ───────────────────────
    DONE               = "done"               # terminal: SUCCESS
    PIPELINE_ERROR     = "pipeline_error"     # terminal: FAILED


@dataclass
class SSEEvent:
    event : SSEEventType
    data  : Dict[str, Any]
    id    : Optional[str] = None

    def serialize(self) -> str:
        """Serializes event to W3C compliant Server-Sent Event format."""
        lines = []
        if self.id:
            lines.append(f"id: {self.id}")
        lines.append(f"event: {self.event.value}")
        lines.append(f"data: {json.dumps(self.data)}")
        lines.append("")
        return "\n".join(lines) + "\n"


def make_success_event(paper_id: str, qa_score: float, detail: Optional[Dict[str, Any]] = None) -> SSEEvent:
    """Creates standard terminal success event."""
    data = {
        "status": "SUCCESS",
        "paper_id": paper_id,
        "qa_score": qa_score,
        "exportable": True,
    }
    if detail:
        data.update(detail)
    return SSEEvent(
        event=SSEEventType.DONE,
        data=data
    )


def make_failure_event(
    code: str,
    stage: str,
    message: str,
    recoverable: bool,
    detail: Optional[Dict[str, Any]] = None
) -> SSEEvent:
    """Creates standard terminal failure event."""
    return SSEEvent(
        event=SSEEventType.PIPELINE_ERROR,
        data={
            "status": "FAILED",
            "code": code,
            "stage": stage,
            "message": message,
            "recoverable": recoverable,
            "detail": detail or {},
        }
    )
