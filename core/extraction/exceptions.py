"""
AION Core Extraction Exceptions — Hard Stop Contracts
=====================================================
Defines ExtractionHardStop replacing all legacy fallbacks as specified in Part III.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ExtractionHardStop(Exception):
    """
    Terminal extraction hard stop exception.
    When raised, legacy fallback extraction is strictly prohibited.
    Qwen is NEVER called after an extraction hard stop.
    """

    def __init__(
        self,
        code: str,
        stage: str,
        message: str,
        recoverable: bool = False,
        detail: Optional[Dict[str, Any]] = None
    ):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.stage = stage
        self.message = message
        self.recoverable = recoverable
        self.detail = detail or {}


# Standard Hard Stop Codes
class HardStopCode:
    TXT_AS_SOURCE_REJECTED         = "TXT_AS_SOURCE_REJECTED"
    EXTRACTION_GATEWAY_UNAVAILABLE = "EXTRACTION_GATEWAY_UNAVAILABLE"
    INSUFFICIENT_VALID_EVIDENCE     = "INSUFFICIENT_VALID_EVIDENCE"
    MODULE_COVERAGE_INSUFFICIENT   = "MODULE_COVERAGE_INSUFFICIENT"
    BINARY_CONTAMINATION_DOMINANT  = "BINARY_CONTAMINATION_DOMINANT"
    ALL_CHUNKS_QUARANTINED         = "ALL_CHUNKS_QUARANTINED"
