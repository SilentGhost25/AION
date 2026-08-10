"""
AION Core Artifacts — Cache Key & Invalidation
===============================================
Implements multi-component CacheKey and stale cache invalidation
as specified in Part VI of the Production Hardening Specification.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.config.dependency_versions import (
    get_docling_version, get_ocr_version, get_pymupdf_version
)
from .manifest import DocumentManifest

logger = logging.getLogger("AION.CacheManager")

AION_EXTRACTION_VERSION = "v1.0.0"


@dataclass
class CacheKey:
    """Multi-component cache key. Cache is invalidated if ANY component changes."""
    source_sha256       : str   # file content hash
    extraction_version  : str   # AION extraction pipeline version
    pymupdf_version     : str   # fitz/pymupdf version
    docling_version     : str   # docling version or "unavailable"
    ocr_version         : str   # tesseract version or "unavailable"
    config_hash         : str = "default_config"   # hash of extraction configuration

    def compute_key(self) -> str:
        components = "|".join([
            self.source_sha256,
            self.extraction_version,
            self.pymupdf_version,
            self.docling_version,
            self.ocr_version,
            self.config_hash,
        ])
        return hashlib.sha256(components.encode()).hexdigest()


def build_cache_key(manifest: DocumentManifest, config_hash: str = "default_config") -> CacheKey:
    return CacheKey(
        source_sha256=manifest.source.sha256,
        extraction_version=AION_EXTRACTION_VERSION,
        pymupdf_version=get_pymupdf_version(),
        docling_version=get_docling_version(),
        ocr_version=get_ocr_version(),
        config_hash=config_hash,
    )


def is_cache_valid(manifest: DocumentManifest, config_hash: str = "default_config") -> bool:
    """Returns True only if cached evidence matches current source and tool versions."""
    evidence_json = manifest.derived.get("evidence_json")
    if not evidence_json:
        return False

    cached_key = getattr(evidence_json, "cache_key", None)
    if not cached_key and isinstance(evidence_json, dict):
        cached_key = evidence_json.get("cache_key")

    if not cached_key:
        return False

    current_key = build_cache_key(manifest, config_hash=config_hash).compute_key()

    if cached_key != current_key:
        logger.info(f"[CACHE] Stale: key mismatch for {manifest.document_id}")
        logger.info(f"[CACHE] Expected: {current_key[:16]}...")
        logger.info(f"[CACHE] Stored:   {cached_key[:16]}...")
        return False

    return True
