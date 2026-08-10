"""
AION Artifact Store — Document Manifest & Artifact Models
==========================================================
Defines immutable SourceArtifacts, derived representation caches, and the single
authoritative DocumentManifest registry record for every document in AION.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class SourceArtifact:
    """
    The immutable, authoritative source file.
    Once written, this record never changes.
    """
    path          : str            # workspace/uploads/{document_id}/original.{ext}
    filename      : str            # original filename from upload
    mime_type     : str            # detected from file header, not extension
    size_bytes    : int
    sha256        : str            # computed on receipt — integrity check
    authoritative : bool = True    # always True for source artifacts

    def __post_init__(self):
        assert self.authoritative, "SourceArtifact must be authoritative"
        assert os.path.exists(self.path), f"Source file not found: {self.path}"
        assert self.sha256, "SHA256 must be computed on receipt"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "authoritative": True,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SourceArtifact:
        return cls(
            path=data["path"],
            filename=data["filename"],
            mime_type=data["mime_type"],
            size_bytes=data["size_bytes"],
            sha256=data["sha256"],
            authoritative=data.get("authoritative", True),
        )


@dataclass
class DerivedArtifact:
    """
    Any representation derived from the source.
    Never authoritative. Never the input to ExtractionGateway.
    """
    path            : str
    derived_type    : str            # "plain_text" | "evidence_json" | "chunks"
    authoritative   : bool = False   # ALWAYS False
    build_timestamp : Optional[str] = None
    source_sha256   : Optional[str] = None   # sha256 of source it was built from

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "derived_type": self.derived_type,
            "authoritative": False,
            "build_timestamp": self.build_timestamp or datetime.now(timezone.utc).isoformat(),
            "source_sha256": self.source_sha256,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DerivedArtifact:
        return cls(
            path=data["path"],
            derived_type=data["derived_type"],
            authoritative=False,
            build_timestamp=data.get("build_timestamp"),
            source_sha256=data.get("source_sha256"),
        )


@dataclass
class DocumentManifest:
    """
    The single registry entry for every uploaded document.
    ArtifactStore reads this — never raw file paths.
    """
    document_id       : str
    created_at        : str
    source            : SourceArtifact
    derived           : Dict[str, DerivedArtifact] = field(default_factory=dict)
    extraction_status : str = "PENDING"      # PENDING | COMPLETE | FAILED
    extraction_report : Optional[Dict[str, Any]] = None

    def get_extraction_source(self) -> str:
        """
        Returns the path that ExtractionGateway must receive.
        Always the original file. Never a derived TXT.
        """
        return self.source.path

    def get_derived_text(self) -> Optional[str]:
        """
        Returns cached plain text path IF it exists.
        Callers must treat this as a cache, not the source.
        """
        d = self.derived.get("plain_text")
        return d.path if d else None

    def is_pdf(self) -> bool:
        return self.source.mime_type == "application/pdf"

    def is_docx(self) -> bool:
        return self.source.mime_type in {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        }

    def invalidate_derived(self):
        """Called when source is re-uploaded or extraction is reset."""
        self.derived.clear()
        self.extraction_status = "PENDING"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "created_at": self.created_at,
            "source": self.source.to_dict(),
            "derived": {k: v.to_dict() for k, v in self.derived.items()},
            "extraction_status": self.extraction_status,
            "extraction_report": self.extraction_report,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DocumentManifest:
        source = SourceArtifact.from_dict(data["source"])
        derived_dict = {
            k: DerivedArtifact.from_dict(v) for k, v in data.get("derived", {}).items()
        }
        return cls(
            document_id=data["document_id"],
            created_at=data["created_at"],
            source=source,
            derived=derived_dict,
            extraction_status=data.get("extraction_status", "PENDING"),
            extraction_report=data.get("extraction_report"),
        )
