"""
AION Core Generation — Qwen Warmup Policy & Sequencing
======================================================
Implements QwenWarmupPolicy ensuring warmup occurs ONLY AFTER evidence gate passes
as specified in Part VIII of the Production Hardening Specification.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger("AION.QwenWarmupPolicy")


class QwenWarmupPolicy:
    """Manages Qwen model warmup sequencing."""

    def __init__(self, qwen_loaded_permanently: bool = True):
        self.qwen_loaded_permanently = qwen_loaded_permanently
        self.is_warmed = qwen_loaded_permanently

    def check_and_warmup(
        self,
        evidence_gate_passed: bool,
        warmup_fn: Optional[Callable[[], Any]] = None
    ) -> bool:
        """
        Executes Qwen warmup strictly after evidence gate passes cleanly.
        If Qwen is permanently loaded in memory, skips per-request warmup.
        """
        if self.qwen_loaded_permanently:
            logger.info("[QWEN] Permanently loaded — skipping per-request warmup")
            return True

        if not evidence_gate_passed:
            logger.warning("[QWEN] Evidence gate failed — Qwen warmup WILL NOT BE EXECUTED")
            return False

        if warmup_fn and not self.is_warmed:
            logger.info("[QWEN] Evidence gate passed — executing Qwen warmup now...")
            try:
                warmup_fn()
                self.is_warmed = True
                logger.info("[QWEN] Qwen warmup completed successfully")
                return True
            except Exception as e:
                logger.error(f"[QWEN] Qwen warmup failed: {e}")
                return False

        return self.is_warmed
