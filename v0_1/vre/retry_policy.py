"""
AION VRE Bounded Retry Controller
=================================
Manages max attempts (N=3) and bounded fallback strategies
(PRIMARY_CHAIN -> ALTERNATE_CHAIN -> TEXT_ONLY).
"""

from __future__ import annotations

from typing import List


class VRERetryController:
    """Bounded Retry Controller preventing infinite recursive retries."""

    MAX_ATTEMPTS: int = 3

    def __init__(self) -> None:
        self.attempts: int = 0
        self.strategies: List[str] = ["PRIMARY_CHAIN", "ALTERNATE_CHAIN", "TEXT_ONLY"]

    def can_retry(self) -> bool:
        return self.attempts < self.MAX_ATTEMPTS

    def next_strategy(self) -> str:
        if self.attempts < len(self.strategies):
            strat = self.strategies[self.attempts]
            self.attempts += 1
            return strat
        self.attempts += 1
        return "TEXT_ONLY"
