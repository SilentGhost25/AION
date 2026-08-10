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
    """Manages derived artifact caches (plain_text, chunks, evidence_json)."""

    @classmethod
    def build_derived_text(cls, document_id: str, store: Optional[ArtifactStore] = None) -> str:
        """Extract plain text from original source file and cache as derived artifact."""
        store = store or ArtifactStore()
        manifest = store.get(document_id)
        source_path = manifest.source.path

        try:
            from core.extraction.gateway import ExtractionGateway
            artifact = ExtractionGateway.extract(source_path, document_id=document_id)
            valid_chunks = [c for c in artifact.chunks if c.is_retrieval_eligible()]
            plain_text = "\n\n".join(c.text for c in valid_chunks)
        except Exception as e:
            logger.warning(f"[CACHE_MANAGER] Gateway extraction for text cache failed ({e}), using raw read")
            plain_text = Path(source_path).read_text(encoding="utf-8", errors="ignore")

        derived = store.store_derived(document_id, "plain_text", plain_text)
        logger.info(f"[CACHE] Derived plain_text built from {source_path} -> {derived.path}")
        return derived.path

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
