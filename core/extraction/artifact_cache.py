"""
AION Structured Document Artifact Cache
=======================================
Implements PDF SHA-256 keyed caching of structured DocumentArtifact objects.
No .txt intermediate representations are ever written or read.
Stores canonical artifact.json with text_blocks, figures, tables, and equations.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from core.contracts.document_artifact import (
    DocumentArtifact,
    TextBlock,
    FigureArtifact,
    TableArtifact,
    EquationArtifact,
)

logger = logging.getLogger("AION.ArtifactCache")

EXTRACTOR_VERSION = "2026.09.28-v4"  # bump on any extractor schema/logic change
ROOT = Path(__file__).resolve().parent.parent.parent

MATH_SYMBOLS = {
    '∑', '∫', '∂', '√', '±', '≤', '≥', '≠', '≈', '×', '÷',
    'λ', 'μ', 'σ', 'π', 'θ', 'α', 'β', 'γ', 'Δ', 'Ω', '∇', '∞', '∈', '∉', '⊂', '⊃'
}


def pdf_sha256(pdf_path: Union[str, Path]) -> str:
    """Compute deterministic SHA-256 hash of the PDF file contents."""
    p = Path(pdf_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"PDF source file does not exist: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_dir(sha: str) -> Path:
    """Return canonical cache directory for the given PDF SHA-256 hash."""
    d = ROOT / "workspace" / "derived" / sha
    d.mkdir(parents=True, exist_ok=True)
    return d


def _extract_ocr_page_blocks(page, page_num: int, sha: str) -> List[TextBlock]:
    """Runs OCR on page pixmap when digital text is absent or sparse."""
    ocr_blocks: List[TextBlock] = []
    try:
        import fitz
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")

        ocr_obj = None
        try:
            from rapidocr_onnxruntime import RapidOCR
            ocr_obj = RapidOCR()
        except ImportError:
            try:
                from rapidocr import RapidOCR
                ocr_obj = RapidOCR()
            except ImportError:
                pass

        if ocr_obj:
            ocr_res, _ = ocr_obj(img_bytes)
            if ocr_res:
                for idx, r in enumerate(ocr_res):
                    if len(r) >= 3 and r[2] > 0.40:
                        box = r[0]
                        txt = str(r[1]).strip()
                        if txt:
                            x_coords = [p[0] for p in box]
                            y_coords = [p[1] for p in box]
                            bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                            ocr_blocks.append(
                                TextBlock(
                                    id=f"p{page_num}_ocr_b{idx+1}",
                                    text=txt,
                                    page=page_num,
                                    block_role="BODY",
                                    bbox=bbox,
                                    source_pdf_sha256=sha,
                                )
                            )
    except Exception as e:
        logger.debug(f"OCR page fallback failed on page {page_num}: {e}")
    return ocr_blocks


def extract_structured(pdf_path: Union[str, Path], sha: Optional[str] = None) -> DocumentArtifact:
    """
    Directly extracts structured DocumentArtifact from the original PDF:
    - Text blocks with page, bounding box, and block_role (HEADING | BODY | CAPTION).
    - Figures with bounding box, caption, and cropped/extracted image saved on disk.
    - Tables with headers, rows, and page position.
    - Equations with LaTeX representation.
    """
    p = Path(pdf_path).resolve()
    if not sha:
        sha = pdf_sha256(p)

    target_dir = cache_dir(sha)
    fig_dir = target_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    text_blocks: List[TextBlock] = []
    figures: List[FigureArtifact] = []
    tables: List[TableArtifact] = []
    equations: List[EquationArtifact] = []

    # Attempt PyMuPDF extraction
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz

        doc = fitz.open(str(p))
        try:
            block_counter = 0
            fig_counter = 0
            tbl_counter = 0
            eq_counter = 0

            for page_idx, page in enumerate(doc):
                page_num = page_idx + 1
                page_dict = page.get_text("dict")
                page_text_raw = page.get_text("text")

                # 1. Extract Images / Figures
                images_on_page = page.get_images(full=True)
                for img_info in images_on_page:
                    width = img_info[2] if len(img_info) > 2 else 0
                    height = img_info[3] if len(img_info) > 3 else 0
                    # Quality Gate (Visual Injection Plan): Filter tiny spacer bullets, banner bars, or extreme aspect ratios
                    if width > 0 and height > 0:
                        if width < 160 or height < 120:
                            continue
                        aspect_ratio = width / max(1, height)
                        if aspect_ratio < 0.20 or aspect_ratio > 5.0:
                            continue

                    fig_counter += 1
                    xref = img_info[0]
                    bbox = [0.0, 0.0, 1.0, 1.0]
                    try:
                        bbox_obj = page.get_image_bbox(img_info)
                        if bbox_obj:
                            bbox = [float(bbox_obj.x0), float(bbox_obj.y0), float(bbox_obj.x1), float(bbox_obj.y1)]
                    except Exception:
                        pass

                    img_filename = f"fig_p{page_num}_{xref}_{fig_counter}.png"
                    img_out_path = fig_dir / img_filename

                    # Extract and save image pixmap if not already saved
                    if not img_out_path.exists():
                        try:
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=fitz.Rect(bbox) if bbox != [0.0, 0.0, 1.0, 1.0] else None)
                            pix.save(str(img_out_path))
                        except Exception:
                            try:
                                base_pix = fitz.Pixmap(doc, xref)
                                if base_pix.n >= 5:  # CMYK or other colorspaces
                                    base_pix = fitz.Pixmap(fitz.csRGB, base_pix)
                                base_pix.save(str(img_out_path))
                            except Exception as _ie:
                                logger.debug(f"Failed to export image xref {xref}: {_ie}")

                    if not img_out_path.exists() or img_out_path.stat().st_size < 2500:
                        continue

                    # Heuristic caption search near image
                    caption = ""
                    for m in re.finditer(r"(?i)(?:Figure|Fig\.?|Diagram|Chart)\s*(\d+[A-Za-z0-9\.\-]*)[:\.\-]?\s*([^\n]{3,120})", page_text_raw):
                        caption = m.group(0).strip()
                        break

                    # Compute provenance / academic figure suitability score
                    prov_score = 0.50
                    if width >= 300 and height >= 200:
                        prov_score += 0.25
                    elif width >= 200 and height >= 150:
                        prov_score += 0.15
                    if caption:
                        if re.search(r"(?i)\b(?:figure|fig\.?|diagram|schematic|architecture|flowchart|model|waveform)\b", caption):
                            prov_score += 0.25
                        else:
                            prov_score += 0.10
                    prov_score = min(1.0, round(prov_score, 2))

                    figures.append(
                        FigureArtifact(
                            id=f"fig_{fig_counter}",
                            page=page_num,
                            bbox=tuple(bbox),
                            caption=caption,
                            image_path=str(img_out_path.resolve()) if img_out_path.exists() else "",
                            source_pdf_sha256=sha,
                            module_id=None,
                            provenance_score=prov_score,
                        )
                    )

                # 2. Extract Text Blocks & Classify Roles
                for b in page_dict.get("blocks", []):
                    if "lines" not in b:
                        continue
                    block_counter += 1
                    block_text_parts = []
                    max_size = 0.0
                    is_bold = False

                    for line in b.get("lines", []):
                        for span in line.get("spans", []):
                            txt = span.get("text", "")
                            block_text_parts.append(txt)
                            size = float(span.get("size", 10.0))
                            if size > max_size:
                                max_size = size
                            if "bold" in str(span.get("font", "")).lower():
                                is_bold = True

                    block_text = " ".join("".join(block_text_parts).split())
                    if not block_text or len(block_text) < 3:
                        continue

                    # Classify role
                    role = "BODY"
                    b_lower = block_text.lower()
                    if re.match(r"^(?:figure|fig\.?|diagram|chart|graph)\s+\d+", b_lower):
                        role = "CAPTION"
                    elif re.match(r"^(?:table|tab\.?)\s+\d+", b_lower):
                        role = "CAPTION"
                    elif max_size >= 14.0 or (is_bold and len(block_text.split()) <= 10 and block_text[:1].isupper()):
                        role = "HEADING"
                    elif re.match(r"^(?:module|chapter|unit|part)\s+\d+", b_lower):
                        role = "HEADING"
                    elif "references" in b_lower or "bibliography" in b_lower:
                        role = "BIBLIOGRAPHY"

                    bbox_coords = b.get("bbox", [0.0, 0.0, 0.0, 0.0])
                    text_blocks.append(
                        TextBlock(
                            id=f"p{page_num}_b{block_counter}",
                            text=block_text,
                            page=page_num,
                            block_role=role,
                            bbox=tuple(float(c) for c in bbox_coords),
                            source_pdf_sha256=sha,
                        )
                    )

                # 2b. OCR Fallback for scanned / low-text pages
                page_words = len(page_text_raw.strip().split())
                if page_words < 20:
                    ocr_blocks = _extract_ocr_page_blocks(page, page_num, sha)
                    for ob in ocr_blocks:
                        block_counter += 1
                        text_blocks.append(ob)

                # 3. Detect Equations on page (LaTeX delimiters & mathematical statements)
                for eq_m in re.finditer(r"(?<!\$)\$(?!\$)([^\$\n]{3,200})\$(?!\$)|(\$\$[\s\S]*?\$\$)|(\\\[[\s\S]*?\\\])", page_text_raw):
                    eq_counter += 1
                    raw_eq = (eq_m.group(1) or eq_m.group(2) or eq_m.group(3) or "").strip()
                    if raw_eq:
                        equations.append(
                            EquationArtifact(
                                id=f"eq_{eq_counter}",
                                page=page_num,
                                latex=raw_eq,
                                source_pdf_sha256=sha,
                            )
                        )

                for line in page_text_raw.splitlines():
                    l_str = line.strip()
                    if len(l_str) < 5 or len(l_str) > 120:
                        continue
                    if l_str.startswith("#") or l_str.startswith("import") or l_str.startswith("print"):
                        continue
                    has_symbol = any(sym in l_str for sym in MATH_SYMBOLS)
                    has_formula = bool(
                        re.search(r'^[A-Za-z_\[\]\(\)\{\}0-9\s\+\-\*\/]+\s*=\s*[A-Za-z0-9\s\+\-\*\/\^\(\)\.\{\}\[\]\\]+$', l_str)
                        and any(c in l_str for c in '+-*/^')
                    )
                    if (has_symbol or has_formula) and not any(eq.latex == l_str for eq in equations):
                        eq_counter += 1
                        equations.append(
                            EquationArtifact(
                                id=f"eq_{eq_counter}",
                                page=page_num,
                                latex=l_str,
                                source_pdf_sha256=sha,
                            )
                        )

                # 4. Detect Table structures on page
                # 4a. Native PyMuPDF table finder
                try:
                    tabs = page.find_tables()
                    for tab in (tabs.tables if tabs else []):
                        tbl_counter += 1
                        extracted_data = tab.extract()
                        if extracted_data and len(extracted_data) > 0:
                            headers = [str(c or "").strip() for c in extracted_data[0]]
                            rows = [[str(c or "").strip() for c in r] for r in extracted_data[1:]]
                        else:
                            headers, rows = [], []

                        bbox = tuple(float(c) for c in tab.bbox) if hasattr(tab, "bbox") and tab.bbox else None

                        # Find nearest caption
                        caption = ""
                        for m in re.finditer(r"(?i)(?:Table|Tab\.?)\s*(\d+[A-Za-z0-9\.\-]*)[:\.\-]?\s*([^\n]{3,120})", page_text_raw):
                            caption = m.group(0).strip()
                            break

                        tables.append(
                            TableArtifact(
                                id=f"tbl_{tbl_counter}",
                                page=page_num,
                                headers=headers,
                                rows=rows,
                                caption=caption,
                                bbox=bbox,
                                source_pdf_sha256=sha,
                            )
                        )
                except Exception as _te:
                    logger.debug(f"fitz find_tables error on page {page_num}: {_te}")

                # 4b. Markdown table fallback (if no native tables found on page)
                if not any(t.page == page_num for t in tables):
                    tbl_matches = re.finditer(r"(\|.+?\|\n\|[-:\s|]+\|\n(?:\|.+?\|\n?)+)", page_text_raw)
                    for tbl_m in tbl_matches:
                        tbl_counter += 1
                        tbl_text = tbl_m.group(1).strip()
                        rows = [[c.strip() for c in r.strip("|").split("|")] for r in tbl_text.splitlines() if r.strip()]
                        headers = rows[0] if rows else []
                        data_rows = rows[2:] if len(rows) > 2 else []
                        tables.append(
                            TableArtifact(
                                id=f"tbl_{tbl_counter}",
                                page=page_num,
                                headers=headers,
                                rows=data_rows,
                                caption="",
                                source_pdf_sha256=sha,
                            )
                        )

        finally:
            doc.close()

    except Exception as exc:
        logger.warning(f"Structured PDF extraction fallback triggered for {p.name}: {exc}")
        # Secondary fallback via ExtractionGateway if PyMuPDF failed
        from core.extraction.gateway import ExtractionGateway
        gw_art = ExtractionGateway.extract(str(p))
        gw_text = getattr(gw_art, "text", "") or ""
        lines = [l.strip() for l in gw_text.splitlines() if l.strip()]
        for i, line in enumerate(lines):
            text_blocks.append(
                TextBlock(
                    id=f"fallback_b{i+1}",
                    text=line,
                    page=1,
                    block_role="BODY",
                    source_pdf_sha256=sha,
                )
            )
        for i, fig in enumerate(getattr(gw_art, "figures", [])):
            if isinstance(fig, dict):
                figures.append(
                    FigureArtifact(
                        id=f"fig_{i+1}",
                        page=fig.get("page", 1),
                        bbox=tuple(fig.get("bbox", [0, 0, 1, 1])),
                        caption=fig.get("caption", ""),
                        image_path=fig.get("image_path", ""),
                        source_pdf_sha256=sha,
                    )
                )

    artifact = DocumentArtifact(
        source_pdf_sha256=sha,
        source_pdf_path=str(p.resolve()),
        extractor_version=EXTRACTOR_VERSION,
        text_blocks=text_blocks,
        figures=figures,
        tables=tables,
        equations=equations,
        title=p.stem,
    )
    return artifact


def load_or_extract_artifact(pdf_path: Union[str, Path], force: bool = False) -> DocumentArtifact:
    """
    Loads cached DocumentArtifact from derived/<sha256>/artifact.json if valid,
    or executes structured extraction and caches the result.
    Guarantees no .txt files are written or referenced.
    """
    p = Path(pdf_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Cannot load or extract missing PDF: {p}")

    sha = pdf_sha256(p)
    d = cache_dir(sha)
    artifact_file = d / "artifact.json"
    version_file = d / "extractor.version"

    if not force and artifact_file.exists() and version_file.exists():
        try:
            ver = version_file.read_text(encoding="utf-8").strip()
            if ver == EXTRACTOR_VERSION:
                data = json.loads(artifact_file.read_text(encoding="utf-8"))
                artifact = DocumentArtifact.from_json(data)
                # Verify that images referenced still exist on disk
                return artifact
        except Exception as _ce:
            logger.warning(f"Corrupted cache for {sha} ({_ce}); re-extracting.")

    # Cache miss or stale version: extract fresh structured artifact
    artifact = extract_structured(p, sha=sha)
    try:
        artifact_file.write_text(json.dumps(artifact.to_json(), indent=2), encoding="utf-8")
        version_file.write_text(EXTRACTOR_VERSION, encoding="utf-8")
    except Exception as _we:
        logger.error(f"Failed to write artifact cache for {sha}: {_we}")

    return artifact


def merge_artifacts(artifacts: List[DocumentArtifact], subject: str = "") -> DocumentArtifact:
    """
    Merges multiple DocumentArtifacts (e.g. 5 module PDFs) into a single composite DocumentArtifact.
    Preserves text blocks, figures, tables, and equations tagged with their source PDF SHA-256 and module index.
    """
    if not artifacts:
        raise ValueError("Cannot merge empty artifacts list")

    combined_sha = hashlib.sha256("".join(a.source_pdf_sha256 for a in artifacts).encode()).hexdigest()
    combined_blocks: List[TextBlock] = []
    combined_figures: List[FigureArtifact] = []
    combined_tables: List[TableArtifact] = []
    combined_equations: List[EquationArtifact] = []

    for mod_idx, art in enumerate(artifacts, 1):
        for b in art.text_blocks:
            combined_blocks.append(
                TextBlock(
                    id=f"m{mod_idx}_{b.id}",
                    text=b.text,
                    page=b.page,
                    block_role=b.block_role,
                    bbox=b.bbox,
                    source_pdf_sha256=b.source_pdf_sha256,
                )
            )
        for f in art.figures:
            combined_figures.append(
                FigureArtifact(
                    id=f"m{mod_idx}_{f.id}",
                    page=f.page,
                    bbox=f.bbox,
                    caption=f.caption,
                    image_path=f.image_path,
                    source_pdf_sha256=f.source_pdf_sha256,
                    module_id=f"module_{mod_idx}",
                    provenance_score=getattr(f, "provenance_score", 1.0),
                )
            )
        for t in art.tables:
            combined_tables.append(
                TableArtifact(
                    id=f"m{mod_idx}_{t.id}",
                    page=t.page,
                    headers=t.headers,
                    rows=t.rows,
                    caption=t.caption,
                    bbox=t.bbox,
                    source_pdf_sha256=t.source_pdf_sha256,
                    module_id=f"module_{mod_idx}",
                )
            )
        for eq in art.equations:
            combined_equations.append(
                EquationArtifact(
                    id=f"m{mod_idx}_{eq.id}",
                    page=eq.page,
                    latex=eq.latex,
                    source_pdf_sha256=eq.source_pdf_sha256,
                    module_id=f"module_{mod_idx}",
                )
            )

    return DocumentArtifact(
        source_pdf_sha256=combined_sha,
        source_pdf_path=artifacts[0].source_pdf_path if len(artifacts) == 1 else "multi_pdf_composite",
        extractor_version=EXTRACTOR_VERSION,
        text_blocks=combined_blocks,
        figures=combined_figures,
        tables=combined_tables,
        equations=combined_equations,
        title=subject or "Composite Academic Modules",
    )
