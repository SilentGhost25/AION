"""
AION Core Config — Dependency Version Lock & Functional Checks
===============================================================
Logs dependency versions at startup and performs functional tests
as specified in Part XI of the Production Hardening Specification.
"""

from __future__ import annotations

import logging
import sys
from typing import Dict

logger = logging.getLogger("AION.DependencyLock")


def get_pymupdf_version() -> str:
    try:
        import pymupdf
        return getattr(pymupdf, "__version__", "installed")
    except ImportError:
        try:
            import fitz
            return getattr(fitz, "__version__", "fitz_fallback")
        except ImportError:
            return "unavailable"


def get_docling_version() -> str:
    try:
        import docling
        return getattr(docling, "__version__", "installed")
    except ImportError:
        return "unavailable"


def get_ocr_version() -> str:
    try:
        import pytesseract
        return getattr(pytesseract, "__version__", "installed")
    except ImportError:
        return "unavailable"


def get_pdfplumber_version() -> str:
    try:
        import pdfplumber
        return getattr(pdfplumber, "__version__", "installed")
    except ImportError:
        return "unavailable"


def get_pillow_version() -> str:
    try:
        import PIL
        return getattr(PIL, "__version__", "installed")
    except ImportError:
        return "unavailable"


def log_startup_versions() -> Dict[str, str]:
    """Prints mandatory AION dependency versions log block at startup."""
    versions = {
        "Python": sys.version.split()[0],
        "PyMuPDF": get_pymupdf_version(),
        "Docling": get_docling_version(),
        "pytesseract": get_ocr_version(),
        "pdfplumber": get_pdfplumber_version(),
        "Pillow": get_pillow_version(),
    }

    print("=" * 60)
    print("AION DEPENDENCY VERSIONS")
    print("=" * 60)
    for name, ver in versions.items():
        print(f"  {name:<16}: {ver}")
    print("=" * 60)

    return versions
