"""
AION Artifact Store — Magic-Byte MIME Detector
===============================================
Detects document MIME type strictly from file content header magic bytes,
never relying on file extension alone. Prevents renaming attacks (e.g. PDF saved as .txt).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger("AION.MIMEDetector")

# File header signatures (magic bytes)
MIME_SIGNATURES: List[Tuple[bytes, int, str]] = [
    (b"%PDF-", 0, "application/pdf"),
    (b"PK\x03\x04", 0, "application/zip"),  # DOCX/XLSX/PPTX are ZIP
    (b"\xff\xd8\xff", 0, "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", 0, "image/png"),
    (b"GIF87a", 0, "image/gif"),
    (b"GIF89a", 0, "image/gif"),
    (b"BM", 0, "image/bmp"),
    (b"II\x2a\x00", 0, "image/tiff"),
    (b"MM\x00\x2a", 0, "image/tiff"),
]


def detect_mime_from_header(file_path: str) -> str:
    """
    Detect MIME type from file content, not filename extension.
    A file saved as .txt that is actually a PDF will be detected as PDF.
    """
    p = Path(file_path)
    if not p.exists():
        return "application/octet-stream"

    try:
        with open(p, "rb") as f:
            header = f.read(64)
    except Exception as e:
        logger.warning(f"[MIME_DETECTOR] Could not read header from {file_path}: {e}")
        return "application/octet-stream"

    for sig_bytes, offset, mime_type in MIME_SIGNATURES:
        if header[offset:offset + len(sig_bytes)] == sig_bytes:
            if mime_type == "application/zip":
                return _detect_office_format(file_path)
            return mime_type

    # Check for plain text (high printable ratio)
    try:
        sample = header.decode("utf-8", errors="strict")
        if all(c.isprintable() or c in "\n\r\t" for c in sample):
            return "text/plain"
    except UnicodeDecodeError:
        pass

    return "application/octet-stream"


def _detect_office_format(file_path: str) -> str:
    """Check ZIP internal structure for Office Open XML formats (DOCX, XLSX, PPTX)."""
    try:
        import zipfile
        with zipfile.ZipFile(file_path, "r") as zf:
            names = zf.namelist()
            if any(n.startswith("word/") for n in names):
                return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            if any(n.startswith("xl/") for n in names):
                return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if any(n.startswith("ppt/") for n in names):
                return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    except Exception:
        pass
    return "application/zip"
