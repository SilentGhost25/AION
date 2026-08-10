"""
AION Artifact Store — Diagnostic Assertions
============================================
Runtime assertions preventing legacy text-only fallback bugs and ensuring
ExtractionGateway receives authoritative source documents.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from .mime_detector import detect_mime_from_header

if TYPE_CHECKING:
    from .resolver import ExtractionSource


def assert_not_txt_source(path: str, context: str = ""):
    """
    Hard assertion: the file being used as an extraction source
    must not be a derived TXT file when original multimodal source exists.
    """
    if path.endswith(".txt") or path.endswith(".txt.gz"):
        mime = detect_mime_from_header(path)
        if mime == "text/plain":
            # Check if a PDF with the same stem exists in workspace
            p_pdf = path[:-4] + ".pdf" if path.endswith(".txt") else path[:-7] + ".pdf"
            if os.path.exists(p_pdf):
                raise AssertionError(
                    f"[{context}] TXT file used as extraction source: {path}\n"
                    f"Authoritative PDF source exists at: {p_pdf}\n"
                    f"Fix: use ArtifactStore.get(document_id).source.path instead of derived text."
                )


def assert_gateway_receives_original(source: "ExtractionSource"):
    """
    Verify the gateway received the actual source file, not a derived cache.
    """
    assert source.manifest.source.authoritative, (
        "ExtractionGateway received a non-authoritative source"
    )
    assert source.path == source.manifest.source.path, (
        f"Gateway path {source.path} != manifest source {source.manifest.source.path}"
    )
    assert os.path.exists(source.path), (
        f"Source file not found: {source.path}"
    )
    if source.manifest.is_pdf():
        assert source.path.endswith(".pdf"), f"Expected PDF path for PDF manifest, got: {source.path}"
