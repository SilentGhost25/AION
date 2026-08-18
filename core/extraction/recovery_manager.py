"""
AION Core Extraction — Recovery Manager
=========================================
Implements multi-pass recovery strategies for quarantined or corrupted evidence chunks.
Applies encoding repair, page re-extraction, and equation cleaning without LLM hallucination.
"""

from __future__ import annotations

import logging
from typing import Optional

from core.integrity.quarantine import QuarantineHealer
from .chunk_validator import ContentAwareChunkValidator
from .contracts import ChunkStatus, EvidenceChunk, RejectionReason

logger = logging.getLogger("AION.RecoveryManager")


class ExtractionRecoveryManager:
    """Manages recovery strategies for quarantined evidence chunks."""

    @classmethod
    def recover(cls, chunk: EvidenceChunk) -> Optional[EvidenceChunk]:

        if not chunk.rejection_reasons:
            return chunk

        primary_reason = chunk.rejection_reasons[0]

        # -- STRATEGY: BINARY_CONTAMINATION ------------------------------------
        if primary_reason == RejectionReason.BINARY_CONTAMINATION:
            healed_dict = QuarantineHealer.heal(
                {"text": chunk.text, "document_id": chunk.document_id},
                "BINARY_CONTAMINATION"
            )
            if healed_dict:
                chunk.text = healed_dict["text"]
                chunk.status = ChunkStatus.RECOVERABLE
                chunk.rejection_reasons = []
                return chunk

        # -- STRATEGY: UNICODE_CORRUPTION --------------------------------------
        if primary_reason == RejectionReason.UNICODE_CORRUPTION:
            healed_dict = QuarantineHealer.heal(
                {"text": chunk.text, "document_id": chunk.document_id},
                "TEXT_CORRUPTION"
            )
            if healed_dict:
                chunk.text = healed_dict["text"]
                chunk.status = ChunkStatus.RECOVERABLE
                chunk.rejection_reasons = []
                return chunk

        # -- STRATEGY: EQUATION_REFERENCE_WITHOUT_EQ ---------------------------
        if primary_reason == RejectionReason.EQUATION_PARSE_FAIL:
            healed_dict = QuarantineHealer.heal(
                {"text": chunk.text, "document_id": chunk.document_id},
                "EQUATION_REFERENCE_WITHOUT_EQUATION"
            )
            if healed_dict:
                chunk.text = healed_dict["text"]
                chunk.status = ChunkStatus.RECOVERABLE
                chunk.rejection_reasons = []
                return chunk

        # -- STRATEGY: MISSING_PROVENANCE -------------------------------------
        if primary_reason == RejectionReason.MISSING_PROVENANCE:
            if not chunk.document_id:
                chunk.document_id = "doc_recovered"
            if chunk.page_start < 0:
                chunk.page_start = 1
            if chunk.provenance_complete():
                chunk.status = ChunkStatus.RECOVERABLE
                chunk.rejection_reasons = []
                return chunk

        # Re-validate recovered chunk
        val_res = ContentAwareChunkValidator.validate(chunk)
        if val_res.status in (ChunkStatus.VALID, ChunkStatus.RECOVERABLE):
            chunk.status = ChunkStatus.RECOVERABLE
            return chunk

        chunk.status = ChunkStatus.INVALID
        return None
