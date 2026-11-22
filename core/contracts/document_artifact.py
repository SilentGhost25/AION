"""
AION Structured Document Artifact Contract
==========================================
Defines the canonical, lossless representation of an extracted academic document.
Preserves text blocks with bounding boxes and roles, figures with captions and image paths,
tables with headers and cells, and equations with LaTeX source.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class AssetIntegrityError(ValueError):
    """Raised when an asset fails cryptographic hash verification or integrity check."""
    pass


DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


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
    image_path: str = ""    # path to extracted image file on disk (workspace-relative or legacy absolute)
    source_pdf_sha256: str = ""
    module_id: Optional[str] = None
    provenance_score: float = 1.0
    image_sha256: str = ""

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
            "image_sha256": self.image_sha256,
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
            image_sha256=str(data.get("image_sha256", "")),
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


def resolve_image_path(
    figure_or_path: Union[FigureArtifact, str, Path],
    workspace_root: Optional[Path] = None,
    verify_hash: bool = True,
) -> Path:
    """
    Authoritatively resolve a visual asset path to an existing, verified file on disk.
    Supports workspace-relative paths, cross-machine absolute path recovery, and hash validation.
    """
    ws_root = Path(workspace_root).resolve() if workspace_root is not None else DEFAULT_WORKSPACE_ROOT

    if isinstance(figure_or_path, FigureArtifact):
        raw_path = figure_or_path.image_path
        expected_hash = figure_or_path.image_sha256
    else:
        raw_path = str(figure_or_path)
        expected_hash = ""

    raw_path = raw_path.strip()
    if not raw_path:
        raise FileNotFoundError("Visual asset path is empty")

    p = Path(raw_path)
    candidate: Optional[Path] = None

    # Case 1: Absolute path that exists directly on this host
    if p.is_absolute() and p.is_file():
        candidate = p
    elif p.is_absolute():
        # Case 2: Host absolute path from another machine (e.g., C:\Users\... or /tmp/...)
        # Attempt recovery by locating recognized relative suffixes under ws_root
        norm = raw_path.replace("\\", "/")
        for suffix_marker in ("workspace/derived/", "derived/", "workspace/figures/", "figures/"):
            if suffix_marker in norm:
                rel_part = norm[norm.index(suffix_marker):]
                cand = (ws_root / rel_part).resolve()
                if cand.is_file():
                    candidate = cand
                    break
        if candidate is None:
            # Check by filename in workspace/figures or workspace/derived
            cand_fig = (ws_root / "workspace" / "figures" / p.name).resolve()
            if cand_fig.is_file():
                candidate = cand_fig
    else:
        # Case 3: Workspace-relative path
        cand1 = (ws_root / p).resolve()
        if cand1.is_file():
            candidate = cand1
        else:
            cand2 = (ws_root / "workspace" / p).resolve()
            if cand2.is_file():
                candidate = cand2

    if candidate is None or not candidate.is_file():
        raise FileNotFoundError(
            f"Visual asset cannot be resolved: '{raw_path}' under workspace '{ws_root}'"
        )

    # Cryptographic hash verification if stored hash is present
    if verify_hash and expected_hash:
        with open(candidate, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        if actual_hash != expected_hash:
            raise AssetIntegrityError(
                f"Asset integrity violation for '{candidate}': expected SHA-256 {expected_hash}, got {actual_hash}"
            )

    return candidate


def validate_physical_image(
    figure_or_path: Union[FigureArtifact, str, Path],
    expected_sha256: Optional[str] = None,
    workspace_root: Optional[Path] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Authoritative physical validation of an image asset:
    - Path resolution
    - File existence & readability
    - Minimum size (> 100 bytes)
    - Magic bytes signature (PNG, JPEG, WebP)
    - Structural decodability (PIL Image.open)
    - Cryptographic SHA-256 match (when hash available)
    """
    try:
        candidate = resolve_image_path(figure_or_path, workspace_root=workspace_root, verify_hash=False)
    except FileNotFoundError as fnf:
        return False, f"FILE_NOT_FOUND: {fnf}"
    except Exception as exc:
        return False, f"RESOLUTION_FAILED: {exc}"

    if not candidate.is_file():
        return False, f"FILE_NOT_FOUND: {candidate}"

    try:
        with open(candidate, "rb") as f:
            data = f.read()
    except Exception as io_err:
        return False, f"UNREADABLE: {io_err}"

    if len(data) <= 100:
        return False, f"CORRUPT_OR_TRIVIAL_SIZE: File is <= 100 bytes ({len(data)} bytes)"

    # Magic byte signature validation
    is_png = data.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = data.startswith(b"\xff\xd8\xff")
    is_webp = data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP"

    if not (is_png or is_jpeg or is_webp):
        return False, f"INVALID_IMAGE_SIGNATURE: Unrecognized file magic bytes ({data[:8]!r})"

    # Structural decodability with PIL
    try:
        from PIL import Image
        with Image.open(candidate) as img:
            img.verify()
    except Exception as dec_err:
        return False, f"STRUCTURALLY_UNREADABLE: {dec_err}"

    # Hash verification
    target_hash = expected_sha256
    if not target_hash and isinstance(figure_or_path, FigureArtifact) and figure_or_path.image_sha256:
        target_hash = figure_or_path.image_sha256

    if target_hash:
        actual_hash = hashlib.sha256(data).hexdigest()
        if actual_hash != target_hash:
            return False, f"HASH_MISMATCH: expected {target_hash}, got {actual_hash}"

    return True, None

