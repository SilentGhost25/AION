# runtime/memory_governor.py

import psutil, gc, logging
from runtime.profiles import MemoryState

LOG = logging.getLogger("aion.memory")


class MemoryGovernor:

    def __init__(self, caution_gb: float = 2.0, critical_gb: float = 1.0):
        self.caution_gb  = caution_gb
        self.critical_gb = critical_gb

    def state(self) -> MemoryState:
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_gb < self.critical_gb:
            return MemoryState.CRITICAL
        if available_gb < self.caution_gb:
            return MemoryState.CAUTION
        return MemoryState.SAFE

    def check_and_act(self) -> MemoryState:
        s = self.state()
        if s == MemoryState.CAUTION:
            LOG.warning(
                f"[MEMORY] CAUTION — available={self._avail():.2f}GB "
                f"< caution={self.caution_gb}GB"
            )
            gc.collect()
        elif s == MemoryState.CRITICAL:
            LOG.error(
                f"[MEMORY] CRITICAL — available={self._avail():.2f}GB "
                f"< critical={self.critical_gb}GB"
            )
            gc.collect()
        return s

    def _avail(self) -> float:
        return psutil.virtual_memory().available / (1024 ** 3)

    def assert_safe_for_generation(self) -> None:
        if self.check_and_act() == MemoryState.CRITICAL:
            raise MemoryError(
                "System memory is critically low. "
                "Generation refused to prevent OOM."
            )
