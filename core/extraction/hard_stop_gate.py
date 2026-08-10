"""
AION Core Extraction — Hard Stop Gate
======================================
Enforces the mandatory minimum viable evidence thresholds.
Blocks downstream question generation and Qwen calls if valid evidence is insufficient.
Returns structured HTTP 422 failure responses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .contracts import RejectionReason
from .reporter import ChunkValidationReport

logger = logging.getLogger("AION.HardStopGate")


@dataclass
class GateDecision:
    action             : str               # "PROCEED" | "BLOCKED"
    reason             : str = ""
    retrieval_eligible : int = 0
    http_payload       : Dict[str, Any] = field(default_factory=dict)


class ExtractionHardStopGate:
    """Hard stop gate enforcing evidence sufficiency before any LLM calls."""

    MIN_VALID_CHUNKS      = 50
    MIN_CHUNKS_PER_MODULE = 5

    @classmethod
    def check(
        cls,
        report: ChunkValidationReport,
        requested_modules: int = 5,
        document_name: str = "uploaded_document"
    ) -> GateDecision:
        eligible = report.get_retrieval_eligible_count()

        # ── CHECK 1: ABSOLUTE MINIMUM ─────────────────────────────────────────
        if eligible < cls.MIN_VALID_CHUNKS:
            report.hard_stop_triggered = True
            report.hard_stop_reason = f"INSUFFICIENT_VALID_EVIDENCE: Retrieval eligible chunks ({eligible}) < MIN_VALID_CHUNKS ({cls.MIN_VALID_CHUNKS})"
            report.recommended_action = "Re-upload the original PDF without TXT conversion."

            cls._log_hard_stop(document_name, report, "INSUFFICIENT_VALID_EVIDENCE")

            payload = cls._build_http_payload(
                document_name,
                report,
                code="INSUFFICIENT_VALID_EVIDENCE",
                message="The uploaded document could not yield enough valid academic chunks.",
            )

            return GateDecision(
                action="BLOCKED",
                reason=report.hard_stop_reason,
                retrieval_eligible=eligible,
                http_payload=payload,
            )

        # ── CHECK 2: PER-MODULE MINIMUM ───────────────────────────────────────
        for mod in range(1, requested_modules + 1):
            mod_chunks = report.per_module_coverage.get(mod, 0)
            if mod_chunks < cls.MIN_CHUNKS_PER_MODULE:
                report.hard_stop_triggered = True
                report.hard_stop_reason = f"Module {mod} has only {mod_chunks} valid chunks (min {cls.MIN_CHUNKS_PER_MODULE})"
                report.recommended_action = f"Ensure document includes coverage for Module {mod}."

                cls._log_hard_stop(document_name, report, f"MODULE_{mod}_INSUFFICIENT_EVIDENCE")

                payload = cls._build_http_payload(
                    document_name,
                    report,
                    code=f"MODULE_{mod}_INSUFFICIENT_EVIDENCE",
                    message=f"Module {mod} has insufficient valid evidence.",
                )

                return GateDecision(
                    action="BLOCKED",
                    reason=report.hard_stop_reason,
                    retrieval_eligible=eligible,
                    http_payload=payload,
                )

        # ── CHECK 3: BINARY CONTAMINATION DOMINANCE ───────────────────────────
        tot = max(report.total_chunks, 1)
        binary_cnt = report.rejection_breakdown.get(RejectionReason.BINARY_CONTAMINATION, 0)
        binary_rate = binary_cnt / tot
        if binary_rate > 0.50:
            report.hard_stop_triggered = True
            report.hard_stop_reason = f"Binary contamination rate ({binary_rate:.1%}) exceeds 50%"
            report.recommended_action = "Re-upload clean original PDF."

            cls._log_hard_stop(document_name, report, "DOCUMENT_BINARY_CONTAMINATION")

            payload = cls._build_http_payload(
                document_name,
                report,
                code="DOCUMENT_BINARY_CONTAMINATION",
                message="Document appears to be corrupted or binary contaminated.",
            )

            return GateDecision(
                action="BLOCKED",
                reason=report.hard_stop_reason,
                retrieval_eligible=eligible,
                http_payload=payload,
            )

        # ALL CHECKS PASS
        return GateDecision(action="PROCEED", retrieval_eligible=eligible)

    @classmethod
    def _log_hard_stop(cls, doc_name: str, report: ChunkValidationReport, root_cause: str):
        logger.error("╔══════════════════════════════════════════════════╗")
        logger.error("║         EXTRACTION HARD STOP                     ║")
        logger.error("╠══════════════════════════════════════════════════╣")
        logger.error(f"║ Document    : {doc_name}")
        logger.error(f"║ Total chunks: {report.total_chunks}")
        logger.error(f"║ Valid        : {report.valid_chunks}")
        logger.error(f"║ Quarantined  : {report.quarantined_chunks}")
        logger.error(f"║ Root cause   : {root_cause}")
        logger.error(f"║ Status       : BLOCKED — Qwen NOT called")
        logger.error("╚══════════════════════════════════════════════════╝")

    @classmethod
    def _build_http_payload(
        cls,
        doc_name: str,
        report: ChunkValidationReport,
        code: str,
        message: str
    ) -> Dict[str, Any]:
        return {
            "status": "blocked",
            "stage": "extraction",
            "code": code,
            "message": message,
            "detail": {
                "document_name": doc_name,
                "total_chunks": report.total_chunks,
                "valid_chunks": report.valid_chunks,
                "retrieval_eligible": report.get_retrieval_eligible_count(),
                "primary_root_cause": report.primary_root_cause.value if report.primary_root_cause else None,
                "rejection_breakdown": {r.value: c for r, c in report.rejection_breakdown.items()},
                "recommended_action": report.recommended_action,
            },
            "recoverable": True,
            "recovery_options": [
                "Upload the original PDF (not a TXT conversion)",
                "Verify the PDF is not password-protected",
                "Verify the PDF is not a scanned image without OCR",
            ],
        }
