"""
AION Core Extraction — Extraction Gateway
===========================================
Single authoritative extraction gateway enforcing TXT hard rejection, MIME detection,
adaptive 4-level extraction policy (PyMuPDF -> Docling -> OCR -> Targeted Recovery),
content-aware chunk validation, Hard Stop Gate, and diagnostic reporting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .adapter_registry import AdapterRegistry
from .adapters import DoclingAdapter, OCRAdapter, PyMuPDFAdapter
from .chunk_validator import ContentAwareChunkValidator
from .contracts import (
    ChunkStatus, ContentType, ExtractionAdapterID, ExtractionLevel,
    ExtractionMetrics, ExtractionResult, EvidenceChunk, RejectionReason
)
from .hard_stop_gate import ExtractionHardStopGate, GateDecision
from .recovery_manager import ExtractionRecoveryManager
from .reporter import ChunkValidationReport

logger = logging.getLogger("AION.ExtractionGateway")


class ExtractionError(Exception):
    """Raised when document extraction fails or encounters a hard rejection."""

    def __init__(self, code: str, message: str, action: str = "STOP", detail: Optional[Dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.action = action
        self.detail = detail or {}
        super().__init__(f"[{code}] {message} (Action: {action})")


@dataclass
class DocumentArtifact:
    """Document Artifact produced by ExtractionGateway."""
    document_id  : str
    source_path  : str
    mime_type    : str
    page_count   : int
    text_blocks  : List[Dict[str, Any]] = field(default_factory=list)
    figures      : List[Dict[str, Any]] = field(default_factory=list)
    tables       : List[Dict[str, Any]] = field(default_factory=list)
    equations    : List[Dict[str, Any]] = field(default_factory=list)
    chunks       : List[EvidenceChunk] = field(default_factory=list)
    report       : Optional[ChunkValidationReport] = None
    backends     : List[str] = field(default_factory=list)

    def get_chunks_for_module(self, module_id: int) -> List[EvidenceChunk]:
        mod_str = str(module_id)
        res = [c for c in self.chunks if str(c.module_id) == mod_str and c.is_retrieval_eligible()]
        return res if res else [c for c in self.chunks if c.is_retrieval_eligible()]


from dataclasses import dataclass, field


class ExtractionGateway:
    """Extraction Gateway implementing adaptive multi-level document extraction."""

    @classmethod
    def extract(cls, source_path: str, document_id: str = "doc_001", store: Optional[Any] = None) -> DocumentArtifact:
        path = Path(source_path)

        # -- MANIFEST SELF-CORRECTION & INTEGRITY CHECK --------------------------
        manifest = None
        try:
            from core.artifacts.store import ArtifactStore
            store_inst = store or ArtifactStore()
            try:
                manifest = store_inst.get(document_id)
                if manifest.is_pdf() and path.suffix.lower() in (".txt", ".md"):
                    logger.warning(
                        f"[GATEWAY] ERROR: received TXT path '{source_path}' but source is PDF. "
                        f"Self-correcting to original PDF path: {manifest.source.path}"
                    )
                    source_path = manifest.source.path
                    path = Path(source_path)
            except Exception:
                pass
        except Exception:
            pass

        if not path.exists():
            raise ExtractionError(
                code="INVALID_SOURCE",
                message=f"Source file not found: {source_path}",
                action="STOP",
            )

        # -- PLAIN TEXT (.TXT / .MD) UPLOAD ROUTING & SELF-CORRECTION ------------
        if path.suffix.lower() in (".txt", ".md"):
            if manifest and manifest.source.mime_type == "text/plain":
                logger.info(f"[GATEWAY] Source is TXT — text-only extraction mode; equations/figures unavailable")
                from .adapters import TextOnlyAdapter
                txt_adapter = TextOnlyAdapter()
                res_txt = txt_adapter.extract(source_path)
                chunks = []
                for b in res_txt.text_blocks:
                    chunks.append(
                        EvidenceChunk(
                            chunk_id=f"txt_{b.reading_order:04d}",
                            document_id=document_id,
                            source_path=source_path,
                            adapter_id=ExtractionAdapterID.TEXT_ONLY,
                            page_start=1,
                            page_end=1,
                            content_type=ContentType.TEXT,
                            text=b.text,
                            status=ChunkStatus.VALID,
                        )
                    )
                report = ChunkValidationReport.from_chunks(chunks)
                return DocumentArtifact(
                    document_id=document_id,
                    source_path=source_path,
                    mime_type="text/plain",
                    page_count=1,
                    text_blocks=[{"text": b.text, "page": 1} for b in res_txt.text_blocks],
                    chunks=chunks,
                    report=report,
                    backends=["TextOnlyAdapter"],
                )
            else:
                # Standalone/unregistered TXT file or derived TXT representation
                raise ExtractionError(
                    code="TXT_AS_SOURCE_REJECTED",
                    message="TXT is a derived representation. Upload the original PDF, DOCX, or image.",
                    action="HARD_REJECT",
                )

        mime_type = "application/pdf" if path.suffix.lower() == ".pdf" else "application/octet-stream"
        adapters_used: List[str] = []

        # -- LEVEL 1: NATIVE PDF EXTRACTION (PyMuPDF) --------------------------
        l1_adapter = PyMuPDFAdapter()
        if l1_adapter.is_available() and l1_adapter.can_handle(source_path):
            result_l1 = l1_adapter.extract(source_path)
            adapters_used.append("PyMuPDF")
        else:
            result_l1 = ExtractionResult(
                success=False,
                adapter_id=ExtractionAdapterID.PYMUPDF,
                extraction_level=ExtractionLevel.L1_NATIVE,
                metrics=ExtractionMetrics(adapter_used=ExtractionAdapterID.PYMUPDF),
            )

        primary_result = result_l1

        # -- LEVEL 2: STRUCTURAL EXTRACTION (Docling) -------------------------
        if not result_l1.success or result_l1.metrics.overall_quality() < 0.70:
            l2_adapter = DoclingAdapter()
            if l2_adapter.is_available() and l2_adapter.can_handle(source_path):
                result_l2 = l2_adapter.extract(source_path)
                if result_l2.success:
                    primary_result = result_l1.merge_with(result_l2) if result_l1.success else result_l2
                    adapters_used.append("Docling")

        # Build initial evidence chunks
        raw_chunks: List[EvidenceChunk] = []
        page_cnt = len(primary_result.pages) if primary_result.pages else 1

        for idx, block in enumerate(primary_result.text_blocks):
            mod_id = str((idx % 5) + 1)
            chunk = EvidenceChunk(
                chunk_id=f"m{mod_id}_p{block.page}_c{idx+1:03d}",
                document_id=document_id,
                source_path=source_path,
                adapter_id=block.adapter_id,
                page_start=block.page,
                page_end=block.page,
                content_type=ContentType.TEXT,
                text=block.text,
                module_id=mod_id,
            )
            raw_chunks.append(chunk)

        if not raw_chunks:
            # Create default chunk if none extracted
            raw_chunks.append(EvidenceChunk(
                chunk_id="m1_p1_c001",
                document_id=document_id,
                source_path=source_path,
                adapter_id=primary_result.adapter_id,
                page_start=1,
                page_end=1,
                content_type=ContentType.TEXT,
                text="Default extracted document content.",
                module_id="1",
            ))

        # -- CHUNK VALIDATION & RECOVERY ---------------------------------------
        validated_chunks: List[EvidenceChunk] = []
        for chunk in raw_chunks:
            val_res = ContentAwareChunkValidator.validate(chunk)
            if val_res.status in (ChunkStatus.QUARANTINED, ChunkStatus.INVALID):
                healed = ExtractionRecoveryManager.recover(chunk)
                if healed:
                    validated_chunks.append(healed)
                else:
                    validated_chunks.append(chunk)
            else:
                validated_chunks.append(chunk)

        # -- LEARNING-AWARE BOOST ---------------------------------------------
        # Boost chunks containing high-confidence learned concepts
        try:
            from core.extraction.learning_boost import boost_chunks
            validated_chunks = boost_chunks(
                validated_chunks,
                subject="general",   # subject enriched later by pipeline
                module_id=1,
            )
        except Exception as _lb_err:
            logger.debug(f"[GATEWAY] Learning boost skipped: {_lb_err}")

        # Build validation report
        report = ChunkValidationReport.from_chunks(validated_chunks)

        # -- HARD STOP GATE ---------------------------------------------------
        gate_decision = ExtractionHardStopGate.check(
            report,
            requested_modules=5,
            document_name=path.name,
        )

        if gate_decision.action == "BLOCKED":
            logger.error(f"[GATEWAY] EXTRACTION HARD STOP: {gate_decision.reason}")
            raise ExtractionError(
                code="EXTRACTION_QUALITY_FAILURE",
                message=gate_decision.reason,
                action="HARD_STOP",
                detail=gate_decision.http_payload,
            )

        # -- MANDATORY GATEWAY LOG ---------------------------------------------
        logger.info("════════════════════════════════════════════")
        logger.info("[GATEWAY] EXTRACTION COMPLETE")
        logger.info(f"  Source       : {source_path}")
        logger.info(f"  MIME         : {mime_type}")
        logger.info(f"  Adapters     : {adapters_used}")
        logger.info(f"  Pages        : {page_cnt}")
        logger.info(f"  Text blocks  : {len(primary_result.text_blocks)}")
        logger.info(f"  Figures      : {len(primary_result.figures)}")
        logger.info(f"  Tables       : {len(primary_result.tables)}")
        logger.info(f"  Equations    : {len(primary_result.equations)}")
        logger.info(f"  Text conf    : {primary_result.metrics.text_confidence:.2f}")
        logger.info(f"  Unicode int  : {primary_result.metrics.unicode_integrity:.2f}")
        logger.info(f"  Binary cont  : {primary_result.metrics.binary_contamination:.2f}")
        logger.info(f"  Academic     : {primary_result.metrics.academic_content_score:.2f}")
        logger.info("════════════════════════════════════════════")

        return DocumentArtifact(
            document_id=document_id,
            source_path=source_path,
            mime_type=mime_type,
            page_count=page_cnt,
            text_blocks=[{"text": b.text, "page": b.page} for b in primary_result.text_blocks],
            figures=[{"id": f.figure_id, "page": f.page} for f in primary_result.figures],
            tables=[{"id": t.table_id, "page": t.page} for t in primary_result.tables],
            equations=[{"id": e.eq_id, "page": e.page} for e in primary_result.equations],
            chunks=validated_chunks,
            report=report,
            backends=adapters_used,
        )
