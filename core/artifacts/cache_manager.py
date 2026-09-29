"""
AION Artifact Store — Derived Cache Manager
=============================================
Manages creation, invalidation, and clearing of derived artifact caches.
Derived artifacts are always built FROM the source, never stored as the source.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from .store import ArtifactStore

logger = logging.getLogger("AION.CacheManager")


class DerivedCacheManager:
    """Manages derived artifact caches (artifact.json, chunks, evidence_json)."""

    @classmethod
    def build_derived_artifact(cls, document_id: str, store: Optional[ArtifactStore] = None) -> str:
        """Extract structured artifact from original source file and cache as derived artifact."""
        store = store or ArtifactStore()
        manifest = store.get(document_id)
        source_path = manifest.source.path

        if Path(source_path).suffix.lower() == ".pdf":
            from core.extraction.artifact_cache import load_or_extract_artifact
            doc_art = load_or_extract_artifact(source_path)
            derived = store.store_derived(document_id, "artifact", doc_art.to_json())
            logger.info(f"[CACHE] Derived structured artifact built from {source_path} -> {derived.path}")
            return derived.path
        else:
            # User-uploaded text or other non-PDF file
            raw_text = Path(source_path).read_text(encoding="utf-8", errors="ignore")
            derived = store.store_derived(document_id, "plain_text", raw_text)
            logger.info(f"[CACHE] Derived text built from user upload {source_path} -> {derived.path}")
            return derived.path

    @classmethod
    def build_derived_text(cls, document_id: str, store: Optional[ArtifactStore] = None) -> str:
        """Backwards compatibility alias for build_derived_artifact."""
        return cls.build_derived_artifact(document_id, store)


    @classmethod
    def invalidate_derived(cls, document_id: str, store: Optional[ArtifactStore] = None):
        """Invalidate derived cache when source is re-uploaded or reset."""
        store = store or ArtifactStore()
        manifest = store.get(document_id)
        manifest.invalidate_derived()
        store.save_manifest(manifest)

        derived_dir = store.derived_dir / document_id
        if derived_dir.exists():
            shutil.rmtree(derived_dir)

        logger.info(f"[CACHE] Derived artifacts invalidated for document {document_id}")
