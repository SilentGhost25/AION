# runtime/governor.py
"""Memory Governor for LAPTOP_FAST profile.

Monitors system RAM and shared GPU memory pressure on Intel Arc iGPU
systems and dynamically adjusts context length, retrieval top-k, and
concurrency to prevent swapping.

States:
    SAFE     — Full parameters (context 4096, top-k 4)
    CAUTION  — Reduced parameters (context 3072, top-k 3)
    CRITICAL — Attempt recovery; if unsuccessful, controlled failure
"""

import gc
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger("aion.governor")


class MemoryState(str, Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    CRITICAL = "CRITICAL"


@dataclass
class MemorySnapshot:
    """Point-in-time memory reading."""

    total_ram_gb: float
    available_ram_gb: float
    used_ram_gb: float
    usage_percent: float
    state: MemoryState


@dataclass
class GovernorRecommendation:
    """Parameter adjustments recommended by the governor."""

    state: MemoryState
    context_length: int
    retrieval_top_k: int
    concurrency: int
    message: str


# ── Thresholds ────────────────────────────────────────────────────────

# Fraction of total RAM usage triggering each state
_CAUTION_THRESHOLD = 0.75  # 75% RAM used → CAUTION
_CRITICAL_THRESHOLD = 0.88  # 88% RAM used → CRITICAL


class MemoryGovernor:
    """Dynamically adjust pipeline parameters based on memory pressure.

    On Intel Arc iGPU systems the GPU shares system RAM, so tracking
    overall RAM usage is a reasonable proxy for combined pressure.
    """

    def __init__(
        self,
        caution_threshold: float = _CAUTION_THRESHOLD,
        critical_threshold: float = _CRITICAL_THRESHOLD,
    ):
        self._caution = caution_threshold
        self._critical = critical_threshold
        self._psutil_available: Optional[bool] = None

    def _check_psutil(self) -> bool:
        if self._psutil_available is not None:
            return self._psutil_available
        try:
            import psutil
            self._psutil_available = True
        except ImportError:
            self._psutil_available = False
        return self._psutil_available

    def snapshot(self) -> MemorySnapshot:
        """Take a memory reading."""
        if self._check_psutil():
            import psutil
            mem = psutil.virtual_memory()
            total = mem.total / (1024**3)
            available = mem.available / (1024**3)
            used = (mem.total - mem.available) / (1024**3)
            pct = mem.percent / 100.0
        else:
            # Fallback: assume 16 GB and moderate usage
            logger.warning("psutil not available — using conservative estimates")
            total = 16.0
            available = 6.0
            used = 10.0
            pct = used / total

        if pct >= self._critical:
            state = MemoryState.CRITICAL
        elif pct >= self._caution:
            state = MemoryState.CAUTION
        else:
            state = MemoryState.SAFE

        return MemorySnapshot(
            total_ram_gb=round(total, 2),
            available_ram_gb=round(available, 2),
            used_ram_gb=round(used, 2),
            usage_percent=round(pct * 100, 1),
            state=state,
        )

    def recommend(self) -> GovernorRecommendation:
        """Evaluate memory state and return parameter recommendations."""
        snap = self.snapshot()

        if snap.state == MemoryState.SAFE:
            return GovernorRecommendation(
                state=MemoryState.SAFE,
                context_length=4096,
                retrieval_top_k=4,
                concurrency=1,
                message=(
                    f"Memory SAFE — {snap.available_ram_gb:.1f} GB available "
                    f"({snap.usage_percent:.0f}% used)"
                ),
            )

        if snap.state == MemoryState.CAUTION:
            # Priority: reduce generation context FIRST, preserve retrieval quality
            # Reducing BM25 top-k hurts question quality more than reducing context
            return GovernorRecommendation(
                state=MemoryState.CAUTION,
                context_length=3072,
                retrieval_top_k=4,  # Preserve retrieval quality
                concurrency=1,
                message=(
                    f"Memory CAUTION — {snap.available_ram_gb:.1f} GB available "
                    f"({snap.usage_percent:.0f}% used). "
                    f"Reducing context to 3072; retrieval preserved."
                ),
            )

        # CRITICAL — reduce everything, attempt recovery
        return GovernorRecommendation(
            state=MemoryState.CRITICAL,
            context_length=2048,
            retrieval_top_k=2,  # Last resort: reduce retrieval
            concurrency=1,
            message=(
                f"Memory CRITICAL — {snap.available_ram_gb:.1f} GB available "
                f"({snap.usage_percent:.0f}% used). "
                f"Attempting recovery."
            ),
        )

    def attempt_recovery(self) -> GovernorRecommendation:
        """Try to free memory and re-evaluate.

        If still CRITICAL after recovery, returns a recommendation
        with context=2048 and top-k=2.  The caller must decide whether
        to proceed or raise a controlled failure.
        """
        logger.info("Governor: attempting memory recovery")

        # Force Python garbage collection
        gc.collect()

        # Re-evaluate
        rec = self.recommend()

        if rec.state == MemoryState.CRITICAL:
            rec.message += " Recovery did not free sufficient memory."
            logger.warning(rec.message)

        return rec


if __name__ == "__main__":
    gov = MemoryGovernor()
    snap = gov.snapshot()
    rec = gov.recommend()
    print("=== Memory Governor ===")
    print(f"  RAM: {snap.total_ram_gb:.1f} GB total, "
          f"{snap.available_ram_gb:.1f} GB available "
          f"({snap.usage_percent:.0f}% used)")
    print(f"  State: {snap.state.value}")
    print(f"  Recommendation: ctx={rec.context_length}, "
          f"top_k={rec.retrieval_top_k}, "
          f"concurrency={rec.concurrency}")
    print(f"  {rec.message}")
