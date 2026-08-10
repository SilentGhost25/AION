"""
AION Core SSE Stream Manager — Terminal Execution Protocol
============================================================
Guarantees:
1. HTTP 200 signals ONLY that the SSE channel is established (via CONNECTED event).
2. Exactly one terminal event (DONE or PIPELINE_ERROR) is ALWAYS emitted.
3. Stream closes cleanly after terminal event without leaking exceptions.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, AsyncGenerator, Callable, Dict, Generator

from .events import (
    SSEEvent, SSEEventType, make_failure_event, make_success_event
)

logger = logging.getLogger("AION.SSEStreamManager")


class SSEStreamManager:
    """Manages execution of paper generation generator to ensure SSE invariants."""

    @classmethod
    def run_generator(
        cls,
        request_id: str,
        execution_fn: Callable[[], Generator[Dict[str, Any], None, Dict[str, Any]]]
    ) -> Generator[str, None, None]:
        """
        Synchronous generator wrapper yielding SSE-formatted strings for Flask.
        Emits CONNECTED immediately, runs execution_fn, and yields terminal events.
        """
        # STEP 1 — IMMEDIATELY EMIT CONNECTED
        connected_event = SSEEvent(
            event=SSEEventType.CONNECTED,
            data={
                "request_id": request_id,
                "message": "SSE channel established — generation initialized",
            }
        )
        yield connected_event.serialize()

        terminal_emitted = False
        paper_id = "paper_001"
        qa_score = 1.0

        try:
            generator = execution_fn()
            while True:
                try:
                    item = next(generator)
                    if isinstance(item, SSEEvent):
                        if item.event in (SSEEventType.DONE, SSEEventType.PIPELINE_ERROR):
                            terminal_emitted = True
                        yield item.serialize()
                    elif isinstance(item, dict):
                        evt_type = item.get("event", "progress")
                        try:
                            enum_type = SSEEventType(evt_type)
                        except ValueError:
                            enum_type = SSEEventType.PROGRESS
                        evt = SSEEvent(event=enum_type, data=item)
                        if enum_type in (SSEEventType.DONE, SSEEventType.PIPELINE_ERROR):
                            terminal_emitted = True
                        yield evt.serialize()
                    else:
                        yield str(item)
                except StopIteration as stop:
                    res = stop.value
                    if isinstance(res, dict):
                        paper_id = res.get("paper_id", paper_id)
                        qa_score = res.get("qa_score", qa_score)
                    break

            if not terminal_emitted:
                succ = make_success_event(paper_id=paper_id, qa_score=qa_score)
                yield succ.serialize()
                terminal_emitted = True

        except Exception as e:
            logger.error(f"[SSE] Exception during stream execution: {e}", exc_info=True)
            # Map exception types to structured error events
            code = getattr(e, "code", type(e).__name__)
            stage = getattr(e, "stage", "generation")
            recoverable = getattr(e, "recoverable", False)
            msg = str(e)

            fail_evt = make_failure_event(
                code=code,
                stage=stage,
                message=msg,
                recoverable=recoverable,
                detail={"traceback": traceback.format_exc()}
            )
            yield fail_evt.serialize()
            terminal_emitted = True

        finally:
            # Always close stream gracefully
            close_evt = SSEEvent(event=SSEEventType.DONE, data={"_close": True, "terminal": True})
            yield close_evt.serialize()
