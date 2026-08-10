"""
AION Artifact Store — Storage Subsystem
========================================
Manages storage of immutable source artifacts, JSON document manifests, and derived
caches. Computes SHA256 integrity hashes and verifies source integrity on lookup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .manifest import DerivedArtifact, DocumentManifest, SourceArtifact
from .mime_detector import detect_mime_from_header

logger = logging.getLogger("AION.ArtifactStore")


class DocumentNotFoundError(Exception):
    """Raised when a requested DocumentManifest is not found."""
    pass


class SourceFileMissingError(Exception):
    """Raised when the authoritative source file is missing from disk."""
    pass


class SourceIntegrityError(Exception):
    """Raised when SHA256 hash of the source file does not match manifest."""
    pass


EXTENSION_MAP = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "text/plain": ".txt",
}


def compute_sha256(file_path: str) -> str:
    """Compute SHA256 hex digest of file content."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class ArtifactStore:
    """Central store for DocumentManifests and source/derived artifacts."""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir:
            self.base_dir = Path(base_dir).resolve()
        else:
            root = Path(__file__).parent.parent.parent.resolve()
            self.base_dir = root / "workspace"

        self.uploads_dir = self.base_dir / "uploads"
        self.derived_dir = self.base_dir / "derived"
        self.manifests_dir = self.base_dir / "manifests"

        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.derived_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

    def store_from_temp(
        self,
        temp_path: str,
        filename: str,
        mime_type: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> DocumentManifest:
        """Store an uploaded file from a temporary location."""
        import uuid

        doc_id = document_id or str(uuid.uuid4())[:8]

        # Detect MIME from content header if not provided
        detected_mime = detect_mime_from_header(temp_path)
        actual_mime = mime_type if (mime_type and mime_type != "application/octet-stream") else detected_mime

        ext = EXTENSION_MAP.get(actual_mime, Path(filename).suffix.lower() or ".bin")
        doc_upload_dir = self.uploads_dir / doc_id
        doc_upload_dir.mkdir(parents=True, exist_ok=True)

        source_path = doc_upload_dir / f"original{ext}"
        shutil.copy2(temp_path, str(source_path))

        sha256_hash = compute_sha256(str(source_path))
        file_size = source_path.stat().st_size

        manifest = DocumentManifest(
            document_id=doc_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            source=SourceArtifact(
                path=str(source_path),
                filename=filename,
                mime_type=actual_mime,
                size_bytes=file_size,
                sha256=sha256_hash,
                authoritative=True,
            ),
            derived={},
            extraction_status="PENDING",
        )

        self.save_manifest(manifest)
        logger.info(f"[STORE] Stored document {doc_id} -> {source_path} ({file_size} bytes, sha256={sha256_hash[:8]})")
        return manifest

    def get(self, document_id: str) -> DocumentManifest:
        """Load manifest and perform integrity checks."""
        manifest_path = self.manifests_dir / f"{document_id}.json"
        if not manifest_path.exists():
            raise DocumentNotFoundError(f"Document manifest not found: {document_id}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        manifest = DocumentManifest.from_dict(data)

        # Integrity Check 1: File existence
        if not os.path.exists(manifest.source.path):
            raise SourceFileMissingError(
                f"Source file missing for document {document_id}: {manifest.source.path}"
            )

        # Integrity Check 2: SHA256 verification
        actual_sha256 = compute_sha256(manifest.source.path)
        if actual_sha256 != manifest.source.sha256:
            raise SourceIntegrityError(
                f"Source integrity failure for {document_id}: expected {manifest.source.sha256}, got {actual_sha256}"
            )

        return manifest

    def store_derived(self, document_id: str, derived_type: str, content: Union[str, bytes, dict]) -> DerivedArtifact:
        """Store a derived artifact (plain_text, chunks, evidence_json)."""
        manifest = self.get(document_id)

        doc_derived_dir = self.derived_dir / document_id
        doc_derived_dir.mkdir(parents=True, exist_ok=True)

        ext = ".json" if isinstance(content, dict) else (".txt" if isinstance(content, str) else ".bin")
        derived_path = doc_derived_dir / f"{derived_type}{ext}"

        if isinstance(content, dict):
            with open(derived_path, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2)
        elif isinstance(content, str):
            derived_path.write_text(content, encoding="utf-8")
        else:
            derived_path.write_bytes(content)

        derived = DerivedArtifact(
            path=str(derived_path),
            derived_type=derived_type,
            authoritative=False,
            build_timestamp=datetime.now(timezone.utc).isoformat(),
            source_sha256=manifest.source.sha256,
        )

        manifest.derived[derived_type] = derived
        self.save_manifest(manifest)
        logger.info(f"[STORE] Derived artifact stored: {derived_type} -> {derived_path}")
        return derived

    def save_manifest(self, manifest: DocumentManifest):
        """Save DocumentManifest to workspace/manifests/{document_id}.json."""
        manifest_path = self.manifests_dir / f"{manifest.document_id}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)
