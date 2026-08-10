"""
AION Core Extraction — Content-Aware Chunk Validator
=====================================================
Validates evidence chunks according to content type (TEXT, EQUATION, TABLE, FIGURE, MIXED).
Applies universal integrity gates (EncodingGate, Provenance, Min Length, PromptSafetyGate).
Routes image-only pages to FIGURE_ONLY for VRE, preventing false rejections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.integrity.encoding_gate import EncodingGate
from core.integrity.equation_gate import EquationIntegrityGate
from core.integrity.prompt_safety_gate import PromptSafetyGate
from .contracts import ChunkStatus, ContentType, EvidenceChunk, RejectionReason

logger = logging.getLogger("AION.ChunkValidator")


@dataclass
class ValidationResult:
    status            : ChunkStatus
    reason            : Optional[RejectionReason] = None
    action            : str = "PASS"   # "PASS" | "QUARANTINE" | "EXCLUDE" | "VRE_ROUTE"
    rejection_reasons : List[RejectionReason] = field(default_factory=list)
    retrieval_penalty : float = 0.0


class ContentAwareChunkValidator:
    """Validates evidence chunks based on their ContentType."""

    MIN_LENGTH_MAP = {
        ContentType.TEXT: 50,
        ContentType.EQUATION: 5,
        ContentType.TABLE: 10,
        ContentType.MIXED: 30,
        ContentType.FIGURE: 0,
        ContentType.HEADER: 0,
        ContentType.FOOTER: 0,
    }

    @classmethod
    def validate(cls, chunk: EvidenceChunk) -> ValidationResult:
        rejection_reasons: List[RejectionReason] = []
        suspicious_count = 0

        # ── STEP 1: CONTENT TYPE ROUTING FOR LAYOUT ARTIFACTS ────────────────
        if chunk.content_type in (ContentType.HEADER, ContentType.FOOTER):
            chunk.status = ChunkStatus.INVALID
            chunk.rejection_reasons = [RejectionReason.EMPTY_CONTENT]
            return ValidationResult(
                status=ChunkStatus.INVALID,
                reason=RejectionReason.EMPTY_CONTENT,
                action="EXCLUDE_FROM_RETRIEVAL_NOT_ERROR",
                rejection_reasons=[RejectionReason.EMPTY_CONTENT],
            )

        if chunk.content_type == ContentType.FIGURE or chunk.status == ChunkStatus.FIGURE_ONLY:
            chunk.status = ChunkStatus.FIGURE_ONLY
            return ValidationResult(
                status=ChunkStatus.FIGURE_ONLY,
                action="ROUTE_TO_VRE_NOT_TEXT_RETRIEVAL",
            )

        # ── STEP 2: UNIVERSAL GATES ──────────────────────────────────────────

        # Gate U1: Binary Contamination (highest priority)
        encoding_report = EncodingGate.analyze(chunk.text)
        if encoding_report.corruption_level in ("BINARY", "CORRUPTED"):
            chunk.status = ChunkStatus.QUARANTINED
            chunk.rejection_reasons = [RejectionReason.BINARY_CONTAMINATION]
            return ValidationResult(
                status=ChunkStatus.QUARANTINED,
                reason=RejectionReason.BINARY_CONTAMINATION,
                action="QUARANTINE",
                rejection_reasons=[RejectionReason.BINARY_CONTAMINATION],
            )

        # Gate U2: Provenance
        if not chunk.provenance_complete():
            chunk.status = ChunkStatus.QUARANTINED
            chunk.rejection_reasons = [RejectionReason.MISSING_PROVENANCE]
            return ValidationResult(
                status=ChunkStatus.QUARANTINED,
                reason=RejectionReason.MISSING_PROVENANCE,
                action="QUARANTINE",
                rejection_reasons=[RejectionReason.MISSING_PROVENANCE],
            )

        # Gate U3: Minimum Length
        min_len = cls.MIN_LENGTH_MAP.get(chunk.content_type, 30)
        if len(chunk.text.strip()) < min_len:
            chunk.status = ChunkStatus.INVALID
            chunk.rejection_reasons = [RejectionReason.EMPTY_CONTENT]
            return ValidationResult(
                status=ChunkStatus.INVALID,
                reason=RejectionReason.EMPTY_CONTENT,
                action="EXCLUDE",
                rejection_reasons=[RejectionReason.EMPTY_CONTENT],
            )

        # ── STEP 3: CONTENT-SPECIFIC VALIDATION ──────────────────────────────

        # TEXT Validation
        if chunk.content_type in (ContentType.TEXT, ContentType.MIXED):
            # Injection scan
            safety = PromptSafetyGate.scan_source(chunk.text)
            if safety.status == "INJECTION_DETECTED":
                chunk.status = ChunkStatus.QUARANTINED
                chunk.rejection_reasons = [RejectionReason.INJECTION_DETECTED]
                return ValidationResult(
                    status=ChunkStatus.QUARANTINED,
                    reason=RejectionReason.INJECTION_DETECTED,
                    action="QUARANTINE",
                    rejection_reasons=[RejectionReason.INJECTION_DETECTED],
                )

            # Printable ratio (math symbols like Σ ∫ ∂ ∇ π are valid)
            nonprintable = sum(1 for c in chunk.text if ord(c) < 32 and c not in "\n\r\t")
            if nonprintable / max(len(chunk.text), 1) > 0.05:
                suspicious_count += 1
                rejection_reasons.append(RejectionReason.UNICODE_CORRUPTION)

        # EQUATION Validation
        if chunk.content_type == ContentType.EQUATION or chunk.has_math():
            if "given formula" in chunk.text.lower() and len(chunk.equation_ids) == 0:
                chunk.status = ChunkStatus.QUARANTINED
                chunk.rejection_reasons = [RejectionReason.EQUATION_PARSE_FAIL]
                return ValidationResult(
                    status=ChunkStatus.QUARANTINED,
                    reason=RejectionReason.EQUATION_PARSE_FAIL,
                    action="QUARANTINE",
                    rejection_reasons=[RejectionReason.EQUATION_PARSE_FAIL],
                )

        # ── STEP 4: FINAL STATUS DECISION ─────────────────────────────────────
        if suspicious_count >= 2:
            chunk.status = ChunkStatus.SUSPICIOUS
            chunk.retrieval_penalty = 0.40
            chunk.rejection_reasons = rejection_reasons
            return ValidationResult(
                status=ChunkStatus.SUSPICIOUS,
                action="PASS_WITH_PENALTY",
                rejection_reasons=rejection_reasons,
                retrieval_penalty=0.40,
            )
        elif suspicious_count == 1:
            chunk.status = ChunkStatus.SUSPICIOUS
            chunk.retrieval_penalty = 0.20
            chunk.rejection_reasons = rejection_reasons
            return ValidationResult(
                status=ChunkStatus.SUSPICIOUS,
                action="PASS_WITH_PENALTY",
                rejection_reasons=rejection_reasons,
                retrieval_penalty=0.20,
            )

        chunk.status = ChunkStatus.VALID
        chunk.retrieval_penalty = 0.0
        return ValidationResult(status=ChunkStatus.VALID, action="PASS")
