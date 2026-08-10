"""
AION VRE Provenance Tracker
===========================
Tracks source provenance records for every visual question.
"""

from __future__ import annotations

from typing import Optional, Tuple
from .contracts import ProvenanceRecord


class ProvenanceTracker:
    """Manages audit records tracing questions back to source document pages."""

    @staticmethod
    def create_record(
        source_document: str,
        page: int,
        figure_id: str,
        module: str,
        concept: str,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        vko_id: str = "",
        operation_chain_id: str = "",
    ) -> ProvenanceRecord:
        record = ProvenanceRecord(
            source_document=source_document,
            page=page,
            figure_id=figure_id,
            module=module,
            concept=concept,
            source_bbox=bbox,
            vko_id=vko_id,
            operation_chain_id=operation_chain_id,
            generated=True,
        )
        record.trace.append(f"Extracted figure '{figure_id}' from page {page}")
        return record

    @staticmethod
    def add_trace(record: ProvenanceRecord, event: str) -> None:
        record.trace.append(event)
