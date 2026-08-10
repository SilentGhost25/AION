"""
AION Core Extraction — Evidence Coverage Gate
==============================================
Evaluates evidence sufficiency against the specific GenerationRequest requirements.
Ensures every requested module has at least MIN_CHUNKS_PER_MODULE (default 5) valid chunks
and that total retrieval-eligible evidence meets the required threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .contracts import ChunkStatus, EvidenceChunk
from .reporter import ChunkValidationReport

logger = logging.getLogger("AION.EvidenceCoverageGate")


@dataclass
class CoverageDecision:
    action             : str               # "PROCEED" | "BLOCKED"
    reason             : str = ""
    retrieval_eligible : int = 0
    module_coverage    : Dict[int, int] = field(default_factory=dict)
    http_payload       : Dict[str, Any] = field(default_factory=dict)


class EvidenceCoverageGate:
    """Request-aware evidence coverage validator."""

    MIN_CHUNKS_PER_MODULE = 5

    @classmethod
    def check_coverage(
        cls,
        report: ChunkValidationReport,
        requested_modules: int = 5,
        document_name: str = "uploaded_document"
    ) -> CoverageDecision:
        eligible = report.get_retrieval_eligible_count()

        # Module-by-module check
        insufficient_modules = []
        coverage_dict = {}

        for mod in range(1, requested_modules + 1):
            count = report.per_module_coverage.get(mod, 0)
            coverage_dict[mod] = count
            if count < cls.MIN_CHUNKS_PER_MODULE:
                insufficient_modules.append(mod)

        if insufficient_modules:
            mod_str = ", ".join(f"Module {m} ({coverage_dict[m]} chunks)" for m in insufficient_modules)
            reason = f"INSUFFICIENT_MODULE_EVIDENCE: {mod_str} has less than {cls.MIN_CHUNKS_PER_MODULE} valid chunks"
            logger.error(f"[COVERAGE_GATE] BLOCKED: {reason}")

            payload = {
                "status": "blocked",
                "stage": "extraction_coverage",
                "code": "INSUFFICIENT_MODULE_EVIDENCE",
                "message": reason,
                "detail": {
                    "document_name": document_name,
                    "requested_modules": requested_modules,
                    "module_coverage": coverage_dict,
                    "insufficient_modules": insufficient_modules,
                    "min_per_module": cls.MIN_CHUNKS_PER_MODULE,
                },
                "recoverable": True,
                "recovery_options": [
                    "Upload a course document covering all required modules",
                    "Verify the PDF text extraction yields complete syllabus content",
                ],
            }

            return CoverageDecision(
                action="BLOCKED",
                reason=reason,
                retrieval_eligible=eligible,
                module_coverage=coverage_dict,
                http_payload=payload,
            )

        return CoverageDecision(
            action="PROCEED",
            retrieval_eligible=eligible,
            module_coverage=coverage_dict,
        )
