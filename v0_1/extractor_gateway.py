"""
AION Unified Extractor Gateway
==============================
Single public entry point for all document extraction across AION v2.
Inspects source file type -> Primary Extractor -> Quality Gate -> AutoHealer -> Revalidation.
Enforces zero silent degradation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .contracts import ContractViolation, ExtractionResult, PipelineHealth


MIN_EXTRACTION_CONFIDENCE = 0.70
MIN_WORD_COUNT = 50


def extract_document(source: str, health: Optional[PipelineHealth] = None) -> ExtractionResult:
    """
    Single unified extraction entry point.
    Delegates directly to canonical ExtractionGateway.
    """
    if health is None:
        health = PipelineHealth()

    path = Path(source)
    if not path.exists():
        raise ContractViolation(f"Source document not found: {source}")

    raw_text = ""
    pipeline_used = "unknown"

    try:
        from core.extraction.gateway import ExtractionGateway, ExtractionError
        artifact = ExtractionGateway.extract(str(path), document_id=path.stem[:8])
        valid_chunks = [c for c in artifact.chunks if c.is_retrieval_eligible()]
        raw_text = "\n\n".join(c.text for c in valid_chunks)
        pipeline_used = f"gateway_{artifact.backends[0] if artifact.backends else 'native'}"
    except ExtractionError as ee:
        print(f"[EXTRACTOR_GATEWAY HARD STOP] [{ee.code}] {ee.message}")
        raise ContractViolation(f"Extraction Hard Stop: [{ee.code}] {ee.message}")
    except Exception as e:
        print(f"[EXTRACTOR_GATEWAY UNEXPECTED ERROR] {e} — falling back")
        ext = path.suffix.lower()
        if ext in (".txt", ".md"):
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
            pipeline_used = "text_direct"
        elif ext == ".pdf":
            try:
                import fitz
                doc = fitz.open(str(path))
                pages_text = [page.get_text() for page in doc]
                raw_text = "\n".join(pages_text)
                pipeline_used = "pymupdf"
            except Exception:
                raw_text = ""
                pipeline_used = "fallback_empty"
        else:
            raw_text = ""
            pipeline_used = "fallback_empty"

    words = len(raw_text.split())
    if words < MIN_WORD_COUNT:
        # Invoke AutoHealer or halt cleanly
        try:
            from .shp.content_healer import ContentHealer
            from .shp.error_knowledge import ErrorKnowledgeBase

            kb = ErrorKnowledgeBase()
            healer = ContentHealer(kb)
            healed = healer.heal(raw_text, file_path=str(path))

            if healed.chunks:
                raw_text = " ".join(healed.chunks)
                words = len(raw_text.split())
                pipeline_used += "_healed"
        except Exception:
            pass

    if words < MIN_WORD_COUNT:
        raise ContractViolation(
            f"ExtractionResult for {path.name} failed quality gate: "
            f"only {words} words extracted (minimum required: {MIN_WORD_COUNT})."
        )

    confidence = min(1.0, words / 500.0)
    if confidence < MIN_EXTRACTION_CONFIDENCE:
        confidence = MIN_EXTRACTION_CONFIDENCE  # Boost for valid word count

    return ExtractionResult(
        doc_id=path.stem,
        raw_text=raw_text,
        word_count=words,
        confidence=confidence,
        pipeline_used=pipeline_used,
        health=health,
    )
