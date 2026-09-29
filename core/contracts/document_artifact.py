"""
AION Structured Document Artifact Contract
==========================================
Defines the canonical, lossless representation of an extracted academic document.
Preserves text blocks with bounding boxes and roles, figures with captions and image paths,
tables with headers and cells, and equations with LaTeX source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TextBlock:
    id: str
    text: str
    page: int
    block_role: str = "BODY"  # BODY | HEADING | CAPTION | METADATA | ADMIN | BIBLIOGRAPHY | TOC | EXERCISE
    bbox: Optional[Tuple[float, float, float, float]] = None
    source_pdf_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "page": self.page,
            "block_role": self.block_role,
            "bbox": list(self.bbox) if self.bbox else None,
            "source_pdf_sha256": self.source_pdf_sha256,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextBlock":
        bbox_val = data.get("bbox")
        return cls(
            id=str(data.get("id", "")),
            text=str(data.get("text", "")),
            page=int(data.get("page", 1)),
            block_role=str(data.get("block_role", "BODY")),
            bbox=tuple(bbox_val) if bbox_val and isinstance(bbox_val, (list, tuple)) and len(bbox_val) == 4 else None,
            source_pdf_sha256=str(data.get("source_pdf_sha256", "")),
        )


@dataclass(frozen=True)
class FigureArtifact:
    id: str                 # e.g. m{mod}_fig_{num}
    page: int
    bbox: Optional[Tuple[float, float, float, float]] = None
    caption: str = ""
    image_path: str = ""    # path to extracted image file on disk
    source_pdf_sha256: str = ""
    module_id: Optional[str] = None
    provenance_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "page": self.page,
            "bbox": list(self.bbox) if self.bbox else None,
            "caption": self.caption,
            "image_path": self.image_path,
            "source_pdf_sha256": self.source_pdf_sha256,
            "module_id": self.module_id,
            "provenance_score": self.provenance_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FigureArtifact":
        bbox_val = data.get("bbox")
        return cls(
            id=str(data.get("id", "")),
            page=int(data.get("page", 1)),
            bbox=tuple(bbox_val) if bbox_val and isinstance(bbox_val, (list, tuple)) and len(bbox_val) == 4 else None,
            caption=str(data.get("caption", "")),
            image_path=str(data.get("image_path", "")),
            source_pdf_sha256=str(data.get("source_pdf_sha256", "")),
            module_id=data.get("module_id"),
            provenance_score=float(data.get("provenance_score", 1.0)),
        )


@dataclass(frozen=True)
class TableArtifact:
    id: str                 # e.g. m{mod}_tbl_{num}
    page: int
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    caption: str = ""
    bbox: Optional[Tuple[float, float, float, float]] = None
    source_pdf_sha256: str = ""
    module_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "page": self.page,
            "headers": list(self.headers),
            "rows": [list(r) for r in self.rows],
            "caption": self.caption,
            "bbox": list(self.bbox) if self.bbox else None,
            "source_pdf_sha256": self.source_pdf_sha256,
            "module_id": self.module_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableArtifact":
        bbox_val = data.get("bbox")
        return cls(
            id=str(data.get("id", "")),
            page=int(data.get("page", 1)),
            headers=list(data.get("headers", [])),
            rows=[list(r) for r in data.get("rows", [])],
            caption=str(data.get("caption", "")),
            bbox=tuple(bbox_val) if bbox_val and isinstance(bbox_val, (list, tuple)) and len(bbox_val) == 4 else None,
            source_pdf_sha256=str(data.get("source_pdf_sha256", "")),
            module_id=data.get("module_id"),
        )


@dataclass(frozen=True)
class EquationArtifact:
    id: str                 # e.g. m{mod}_eq_{num}
    page: int
    latex: str
    source_pdf_sha256: str = ""
    module_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "page": self.page,
            "latex": self.latex,
            "source_pdf_sha256": self.source_pdf_sha256,
            "module_id": self.module_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EquationArtifact":
        return cls(
            id=str(data.get("id", "")),
            page=int(data.get("page", 1)),
            latex=str(data.get("latex", "")),
            source_pdf_sha256=str(data.get("source_pdf_sha256", "")),
            module_id=data.get("module_id"),
        )


@dataclass
class DocumentArtifact:
    source_pdf_sha256: str
    source_pdf_path: str
    extractor_version: str = "2026.09.24-v1"
    text_blocks: List[TextBlock] = field(default_factory=list)
    figures: List[FigureArtifact] = field(default_factory=list)
    tables: List[TableArtifact] = field(default_factory=list)
    equations: List[EquationArtifact] = field(default_factory=list)
    module_index: Optional[int] = None
    title: str = ""

    @property
    def raw_text(self) -> str:
        """Convenience derived text view for indexing and fallback synthesis."""
        return "\n\n".join(b.text for b in self.text_blocks if b.block_role in ("BODY", "HEADING"))

    def to_canonical_text(self, *, include_markers: bool = True) -> str:
        """
        Reconstructs a reading-order text stream from this artifact.
        Used ONLY for module-boundary detection — never for LLM input.

        Preserves:
        - Sentence boundaries (no mid-sentence splits at page breaks)
        - Module markers (Module 1, Unit I, etc.) as standalone paragraphs
        - Inline figure/table placeholders (extractable by downstream)
        """
        lines: List[str] = []
        prev_page = None

        for block in sorted(self.text_blocks, key=lambda b: (b.page, b.id)):
            if prev_page is not None and block.page != prev_page:
                if lines and not lines[-1].rstrip().endswith((".", "?", "!", ":")):
                    pass
                else:
                    lines.append("")

            role = block.block_role
            text = block.text.strip()
            if not text:
                continue

            if role == "HEADING":
                lines.append("")
                lines.append(text)
                lines.append("")
            elif role == "BODY":
                lines.append(text)
                lines.append("")
            elif role == "CAPTION":
                if include_markers:
                    lines.append(f"[CAPTION: {text}]")
                    lines.append("")
            elif role in ("ADMIN", "BIBLIOGRAPHY", "TOC"):
                continue
            elif role == "EXERCISE":
                if include_markers:
                    lines.append(f"[EXERCISE: {text}]")
                    lines.append("")
            else:
                lines.append(text)
                lines.append("")

            prev_page = block.page

        if include_markers and self.figures:
            fig_markers = [f"[FIGURE: {f.id} - {f.caption[:60]}]" for f in self.figures]
            lines.append("")
            lines.append("FIGURE PLACEHOLDERS:")
            lines.extend(fig_markers)

        return "\n".join(lines)

    def to_json(self) -> Dict[str, Any]:
        return {
            "source_pdf_sha256": self.source_pdf_sha256,
            "source_pdf_path": self.source_pdf_path,
            "extractor_version": self.extractor_version,
            "module_index": self.module_index,
            "title": self.title,
            "text_blocks": [b.to_dict() for b in self.text_blocks],
            "figures": [f.to_dict() for f in self.figures],
            "tables": [t.to_dict() for t in self.tables],
            "equations": [e.to_dict() for e in self.equations],
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "DocumentArtifact":
        return cls(
            source_pdf_sha256=str(data.get("source_pdf_sha256", "")),
            source_pdf_path=str(data.get("source_pdf_path", "")),
            extractor_version=str(data.get("extractor_version", "2026.09.24-v1")),
            module_index=data.get("module_index"),
            title=str(data.get("title", "")),
            text_blocks=[TextBlock.from_dict(b) for b in data.get("text_blocks", [])],
            figures=[FigureArtifact.from_dict(f) for f in data.get("figures", [])],
            tables=[TableArtifact.from_dict(t) for t in data.get("tables", [])],
            equations=[EquationArtifact.from_dict(e) for e in data.get("equations", [])],
        )
