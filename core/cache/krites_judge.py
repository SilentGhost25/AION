"""
Krites-style LLM judge for async cache promotion.

When a query lands in the "borderline" similarity band — close to
the static threshold but below it — the judge asks an LLM whether the
existing static response is acceptable for the new query. Approval
promotes the static response into the dynamic tier, expanding the
cache's effective reach without adding prompt-time latency.

Deterministic under test:
    - The LLM call is injected as `judge_fn`
    - Time is injected as `clock`
    - Rate limiting is enforced per document

Non-responsibilities:
    - Scheduling (the cache owns the executor)
    - Persistence (the cache writes entries)
    - Prompt construction (the caller supplies judge_fn)
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Callable, Dict


JudgeFn = Callable[[str, dict, str], bool]
"""
Signature: (static_query, static_response, new_query) -> approved?
"""


class KritesJudge:
    """Rate-limited wrapper around an LLM-based cache promotion judge."""

    def __init__(
        self,
        judge_fn: JudgeFn,
        throttle_per_min: int = 10,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if throttle_per_min < 1:
            raise ValueError("throttle_per_min must be >= 1")
        self._judge_fn = judge_fn
        self._throttle = throttle_per_min
        self._clock = clock
        self._lock = threading.Lock()
        self._calls: Dict[str, deque] = defaultdict(deque)

    def should_judge(self, doc_hash: str) -> bool:
        """Return True if the per-doc throttle window allows a call."""
        with self._lock:
            log = self._calls[doc_hash]
            now = self._clock()
            cutoff = now - 60.0
            while log and log[0] < cutoff:
                log.popleft()
            return len(log) < self._throttle

    def judge(
        self,
        *,
        doc_hash: str,
        static_query: str,
        static_response: dict,
        new_query: str,
    ) -> bool:
        """
        Ask the LLM whether the static response is acceptable for the new query.

        Always returns bool. Exceptions from judge_fn are logged and
        treated as rejections — the caller must never crash on judge failure.
        """
        with self._lock:
            self._calls[doc_hash].append(self._clock())

        try:
            return bool(self._judge_fn(static_query, static_response, new_query))
        except Exception as e:
            print(f"[KRITES] judge_fn raised for doc={doc_hash}: {e}", flush=True)
            return False

    def snapshot(self) -> Dict[str, int]:
        """Return call counts per document over the last 60s."""
        with self._lock:
            now = self._clock()
            cutoff = now - 60.0
            return {
                doc: sum(1 for t in log if t >= cutoff)
                for doc, log in self._calls.items()
            }
