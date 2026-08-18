# runtime/watchdog.py
"""Pipeline watchdog with phase-aware budget allocation and per-slot timeouts.

Budget breakdown:
    600s HARD_DEADLINE (total)
    540s TARGET (presentation-safe)
    ├-- dataset_discovery   5s
    ├-- extraction         60s
    ├-- indexing            15s
    ├-- planning            5s
    ├-- generation        390s  (per-slot: ~45s)
    ├-- validation         40s
    └-- assembly_export    25s

If a slot exceeds its budget:
    1. Targeted retry with shorter output
    2. If timeout again -> block slot
    3. Never allow one bad inference to consume the entire paper's budget
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from runtime.profiles.base import TimeoutBudget

logger = logging.getLogger("aion.watchdog")


class Phase(str, Enum):
    DATASET_DISCOVERY = "dataset_discovery"
    EXTRACTION = "extraction"
    INDEXING = "indexing"
    PLANNING = "planning"
    GENERATION = "generation"
    VALIDATION = "validation"
    ASSEMBLY_EXPORT = "assembly_export"


class TimeoutError(RuntimeError):
    """Raised when a phase or slot exceeds its budget."""


@dataclass
class PhaseTimer:
    """Tracks elapsed time for a single pipeline phase."""

    phase: Phase
    budget_seconds: float
    start_time: float = 0.0
    end_time: float = 0.0
    completed: bool = False

    @property
    def elapsed(self) -> float:
        if self.completed:
            return self.end_time - self.start_time
        if self.start_time > 0:
            return time.perf_counter() - self.start_time
        return 0.0

    @property
    def remaining(self) -> float:
        return max(0.0, self.budget_seconds - self.elapsed)

    @property
    def exceeded(self) -> bool:
        return self.elapsed > self.budget_seconds


@dataclass
class SlotTimer:
    """Tracks elapsed time for a single generation slot."""

    slot_id: str
    budget_seconds: float
    attempt: int = 1
    start_time: float = 0.0
    end_time: float = 0.0
    completed: bool = False
    blocked: bool = False

    @property
    def elapsed(self) -> float:
        if self.completed:
            return self.end_time - self.start_time
        if self.start_time > 0:
            return time.perf_counter() - self.start_time
        return 0.0

    @property
    def exceeded(self) -> bool:
        return self.elapsed > self.budget_seconds


class PipelineWatchdog:
    """Manages the global deadline and per-phase/per-slot budgets."""

    def __init__(self, budget: Optional[TimeoutBudget] = None):
        self._budget = budget or TimeoutBudget()
        self._global_start: float = 0.0
        self._phases: Dict[Phase, PhaseTimer] = {}
        self._slots: Dict[str, SlotTimer] = {}

        # Initialise phase timers from budget
        phase_budgets = {
            Phase.DATASET_DISCOVERY: self._budget.dataset_discovery,
            Phase.EXTRACTION: self._budget.extraction,
            Phase.INDEXING: self._budget.indexing,
            Phase.PLANNING: self._budget.planning,
            Phase.GENERATION: self._budget.generation,
            Phase.VALIDATION: self._budget.validation,
            Phase.ASSEMBLY_EXPORT: self._budget.assembly_export,
        }
        for phase, secs in phase_budgets.items():
            self._phases[phase] = PhaseTimer(phase=phase, budget_seconds=secs)

    # -- Global Timer --------------------------------------------------

    def start(self) -> None:
        """Start the global pipeline timer."""
        self._global_start = time.perf_counter()
        logger.info(
            f"Watchdog started — HARD_DEADLINE={self._budget.hard_deadline:.0f}s, "
            f"TARGET={self._budget.target:.0f}s"
        )

    @property
    def global_elapsed(self) -> float:
        if self._global_start == 0:
            return 0.0
        return time.perf_counter() - self._global_start

    @property
    def global_remaining(self) -> float:
        return max(0.0, self._budget.hard_deadline - self.global_elapsed)

    @property
    def target_remaining(self) -> float:
        return max(0.0, self._budget.target - self.global_elapsed)

    @property
    def hard_deadline_exceeded(self) -> bool:
        return self.global_elapsed > self._budget.hard_deadline

    @property
    def target_exceeded(self) -> bool:
        return self.global_elapsed > self._budget.target

    def check_global(self) -> None:
        """Raise TimeoutError if the hard deadline has been exceeded."""
        if self.hard_deadline_exceeded:
            raise TimeoutError(
                f"HARD_DEADLINE exceeded: {self.global_elapsed:.1f}s > "
                f"{self._budget.hard_deadline:.0f}s"
            )

    # -- Phase Timer ---------------------------------------------------

    def start_phase(self, phase: Phase) -> PhaseTimer:
        """Begin tracking a pipeline phase."""
        self.check_global()
        timer = self._phases[phase]
        timer.start_time = time.perf_counter()
        timer.completed = False
        logger.info(f"Phase {phase.value} started (budget: {timer.budget_seconds:.0f}s)")
        return timer

    def end_phase(self, phase: Phase) -> PhaseTimer:
        """Mark a phase as completed."""
        timer = self._phases[phase]
        timer.end_time = time.perf_counter()
        timer.completed = True
        status = "OK" if not timer.exceeded else "OVER BUDGET"
        logger.info(
            f"Phase {phase.value} completed in {timer.elapsed:.1f}s "
            f"[{status}]"
        )
        return timer

    def check_phase(self, phase: Phase) -> None:
        """Check if a phase has exceeded its budget."""
        self.check_global()
        timer = self._phases[phase]
        if timer.exceeded:
            raise TimeoutError(
                f"Phase {phase.value} exceeded budget: "
                f"{timer.elapsed:.1f}s > {timer.budget_seconds:.0f}s"
            )

    # -- Slot Timer ----------------------------------------------------

    def start_slot(self, slot_id: str, attempt: int = 1) -> SlotTimer:
        """Begin tracking a generation slot."""
        self.check_global()
        timer = SlotTimer(
            slot_id=slot_id,
            budget_seconds=self._budget.per_slot,
            attempt=attempt,
            start_time=time.perf_counter(),
        )
        self._slots[f"{slot_id}_a{attempt}"] = timer
        return timer

    def end_slot(self, slot_id: str, attempt: int = 1) -> SlotTimer:
        """Mark a slot generation as completed."""
        key = f"{slot_id}_a{attempt}"
        timer = self._slots.get(key)
        if timer:
            timer.end_time = time.perf_counter()
            timer.completed = True
        return timer

    def check_slot(self, slot_id: str, attempt: int = 1) -> None:
        """Check if a slot has exceeded its budget."""
        self.check_global()
        key = f"{slot_id}_a{attempt}"
        timer = self._slots.get(key)
        if timer and timer.exceeded:
            raise TimeoutError(
                f"Slot {slot_id} (attempt {attempt}) exceeded budget: "
                f"{timer.elapsed:.1f}s > {timer.budget_seconds:.0f}s"
            )

    def block_slot(self, slot_id: str) -> None:
        """Mark a slot as permanently blocked after exhausting retries."""
        for key, timer in self._slots.items():
            if key.startswith(slot_id):
                timer.blocked = True
        logger.error(f"Slot {slot_id} BLOCKED — all retries exhausted")

    # -- Status Report -------------------------------------------------

    def status(self) -> dict:
        """Return a summary of all timers for display."""
        return {
            "global_elapsed": round(self.global_elapsed, 1),
            "global_remaining": round(self.global_remaining, 1),
            "target_remaining": round(self.target_remaining, 1),
            "hard_deadline_exceeded": self.hard_deadline_exceeded,
            "target_exceeded": self.target_exceeded,
            "phases": {
                phase.value: {
                    "elapsed": round(timer.elapsed, 1),
                    "budget": timer.budget_seconds,
                    "exceeded": timer.exceeded,
                    "completed": timer.completed,
                }
                for phase, timer in self._phases.items()
            },
            "slots": {
                key: {
                    "elapsed": round(timer.elapsed, 1),
                    "budget": timer.budget_seconds,
                    "attempt": timer.attempt,
                    "exceeded": timer.exceeded,
                    "blocked": timer.blocked,
                }
                for key, timer in self._slots.items()
            },
        }
