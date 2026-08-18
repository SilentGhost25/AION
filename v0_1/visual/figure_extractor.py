"""
AION Visual: Figure Extractor
Extracts images from PDF and DOCX with full provenance.
"""

from __future__ import annotations

import re
import os
import hashlib
from pathlib import Path
from typing import Optional

from .figure_card import FigureCard, FigureRegistry, VisualFact


def _clean_pdf_artifacts(text: str) -> str:
    """Remove residual [PDF p.N] markers from extracted text."""
    if not text:
        return ""
    text = re.sub(r"\[PDF\s+p\.\d+\]", "", text, flags=re.I)
    return text.strip()


# -- Visual type classifier (rule-based, no ML needed) --------

_VISUAL_TYPE_RULES = {
    "flowchart":    re.compile(
        r"\b(flow|process|step|decision|loop|arrow|sequence|workflow)\b", re.I
    ),
    "architecture": re.compile(
        r"\b(layer|block|module|component|architecture|system|network|node|"
        r"encoder|decoder|input|output|pipeline)\b", re.I
    ),
    "graph":        re.compile(
        r"\b(axis|axes|x-axis|y-axis|plot|curve|trend|graph|chart|bar|"
        r"histogram|scatter|line|legend)\b", re.I
    ),
    "circuit":      re.compile(
        r"\b(resistor|capacitor|transistor|gate|voltage|current|signal|"
        r"circuit|amplifier|diode|logic)\b", re.I
    ),
    "table":        re.compile(
        r"\b(row|column|cell|table|entry|value|field)\b", re.I
    ),
    "equation":     re.compile(
        r"[=∑∫∂∇√±×÷]|\\frac|\\sum|\\int|\bequation\b", re.I
    ),
    "code":         re.compile(
        r"\b(def |class |import |function|return|void|int |char )\b", re.I
    ),
    "photo":        re.compile(
        r"\b(photograph|image|photo|micrograph|specimen|sample|"
        r"experiment|apparatus)\b", re.I
    ),
}

_CAPTION_RE = re.compile(
    r"^(Fig(?:ure)?\.?\s*\d+[\.\d]*"
    r"|Diagram\s*\d+"
    r"|Image\s*\d+"
    r"|Chart\s*\d+"
    r"|Graph\s*\d+"
    r"|Plate\s*\d+"
    r"|Exhibit\s*\d+)",
    re.I
)

_REF_RE = re.compile(
    r"\b(Fig(?:ure)?\.?\s*\d+[\.\d]*"
    r"|Diagram\s*\d+"
    r"|Image\s*\d+"
    r"|shown\s+(?:in|below|above)"
    r"|refer(?:red)?\s+to\s+(?:the\s+)?(?:figure|diagram|image))",
    re.I
)

# Images below this pixel area are likely decorative
_MIN_IMAGE_AREA = 80 * 80


def _classify_visual_type(text: str) -> str:
    for vtype, pat in _VISUAL_TYPE_RULES.items():
        if pat.search(text):
            return vtype
    return "unknown"


def _is_decorative(width: int, height: int, img_bytes: bytes) -> bool:
    """Reject tiny, blank, or repeated decorative images."""
    if width * height < _MIN_IMAGE_AREA:
        return True
    if len(set(img_bytes[:200])) < 5:
        return True
    return False


def _compute_provenance_score(card: FigureCard) -> float:
    """
    Higher score = more reliable for placement.
    Based purely on available evidence.
    """
    score = 0.0
    if card.explicit_refs:
        score += 0.50
    if card.caption:
        score += 0.25
    if card.ocr_text and len(card.ocr_text.split()) > 3:
        score += 0.10
    if card.section_title:
        score += 0.10
    if card.visual_type != "unknown":
        score += 0.05
    return round(min(score, 1.0), 3)


# -- PDF Extractor ---------------------------------------------

def _extract_from_pdf(
    path:        Path,
    doc_id:      str,
    module_map:  dict[int, str],
    asset_dir:   Path,
    image_url_prefix: str,
) -> list[FigureCard]:
    try:
        import fitz
    except ImportError:
        print("[FIGURE] PyMuPDF not installed — PDF figure extraction skipped")
        return []

    doc    = fitz.open(str(path))
    cards  = []
    fig_no = 0

    for page_num, page in enumerate(doc, 1):
        module_id = module_map.get(page_num, "module_1")

        raw_blocks = page.get_text("dict")["blocks"]
        text_blocks = []
        for b in raw_blocks:
            if b.get("type") != 0:
                continue
            lines = b.get("lines", [])
            text  = " ".join(
                span["text"]
                for ln in lines
                for span in ln.get("spans", [])
            ).strip()
            if text:
                text_blocks.append({
                    "text": text,
                    "y0":   b["bbox"][1],
                    "y1":   b["bbox"][3],
                })

        all_page_text = " ".join(b["text"] for b in text_blocks)
        page_refs = _REF_RE.findall(all_page_text)

        img_list = page.get_images(full=True)

        for img_order, img_info in enumerate(img_list):
            xref = img_info[0]
            try:
                base_img   = doc.extract_image(xref)
                img_bytes  = base_img["image"]
                img_ext    = base_img.get("ext", "png")
                img_width  = base_img.get("width",  0)
                img_height = base_img.get("height", 0)

                if _is_decorative(img_width, img_height, img_bytes):
                    continue

                fig_no  += 1
                fig_id   = f"doc_{doc_id}_p{page_num:03d}_f{fig_no:02d}"
                filename = f"{fig_id}.{img_ext}"
                img_path = asset_dir / filename
                img_path.write_bytes(img_bytes)

                rects = page.get_image_rects(xref)
                bbox  = rects[0] if rects else None

                preceding, following, caption = _get_surrounding_text(
                    text_blocks, bbox
                )

                ocr_text = _quick_ocr(img_bytes)
                section_title = _nearest_heading(text_blocks, bbox)

                combined  = f"{caption} {ocr_text} {preceding} {following}"
                vtype     = _classify_visual_type(combined)

                explicit = [
                    ref for ref in page_refs
                    if any(
                        part in caption
                        for part in str(ref).split()
                        if len(part) > 2
                    )
                ] or (page_refs if page_refs else [])

                card = FigureCard(
                    id               = fig_id,
                    document_id      = doc_id,
                    module_id        = module_id,
                    page             = page_num,
                    figure_index     = fig_no,
                    image_path       = str(img_path),
                    image_url        = f"{image_url_prefix}/{filename}",
                    caption          = caption,
                    ocr_text         = ocr_text,
                    preceding_text   = preceding,
                    following_text   = following,
                    section_title    = section_title,
                    explicit_refs    = [str(r) for r in explicit],
                    visual_type      = vtype,
                )

                card.provenance_score = _compute_provenance_score(card)

                if not caption and not ocr_text and not explicit:
                    card.eligible    = False
                    card.skip_reason = "no_evidence"

                cards.append(card)

            except Exception as e:
                print(f"[FIGURE] Skipping image xref={xref}: {e}")
                continue

    doc.close()
    print(f"[FIGURE] PDF: extracted {len(cards)} eligible figures from {path.name}")
    return cards


def _get_surrounding_text(
    text_blocks: list[dict],
    bbox,
    chars: int = 400,
) -> tuple[str, str, str]:
    if bbox is None:
        return "", "", ""

    img_y0 = bbox.y0
    img_y1 = bbox.y1

    above, below = [], []
    for b in text_blocks:
        if b["y1"] <= img_y0:
            above.append((b["y1"], b["text"]))
        elif b["y0"] >= img_y1:
            below.append((b["y0"], b["text"]))

    above.sort(key=lambda x: -x[0])
    below.sort(key=lambda x:  x[0])

    caption = ""
    for _, text in below[:4]:
        if _CAPTION_RE.match(text.strip()):
            caption = text.strip()
            break
    if not caption:
        for _, text in above[:2]:
            if _CAPTION_RE.match(text.strip()):
                caption = text.strip()
                break

    preceding = " ".join(t for _, t in above[:3])[-chars:]
    following = " ".join(t for _, t in below[:3])[:chars:]

    return preceding.strip(), following.strip(), caption


def _nearest_heading(
    text_blocks: list[dict],
    bbox,
) -> str:
    if bbox is None:
        return ""

    heading_re = re.compile(
        r"^\d+[\.\d]*\s+[A-Z]|^[A-Z][A-Z\s]{4,}$|"
        r"^(Chapter|Module|Section|Unit)\s+\d+",
        re.M
    )
    img_y0 = bbox.y0
    candidates = []
    for b in text_blocks:
        if b["y1"] <= img_y0 and heading_re.match(b["text"].strip()):
            candidates.append((b["y1"], b["text"].strip()))

    if candidates:
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]
    return ""


def _quick_ocr(img_bytes: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(img_bytes))
        return pytesseract.image_to_string(img, timeout=5).strip()
    except Exception:
        pass

    try:
        from rapidocr_onnxruntime import RapidOCR
        import numpy as np
        from PIL import Image
        import io
        img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        arr   = np.array(img)
        ocr   = RapidOCR()
        result, _ = ocr(arr)
        if result:
            return " ".join(r[1] for r in result).strip()
    except Exception:
        pass

    return ""


# -- DOCX Extractor -------------------------------------------

def _extract_from_docx(
    path:        Path,
    doc_id:      str,
    module_map:  dict[int, str],
    asset_dir:   Path,
    image_url_prefix: str,
) -> list[FigureCard]:
    try:
        import zipfile
        from lxml import etree
    except ImportError:
        print("[FIGURE] lxml not installed — DOCX figure extraction skipped")
        return []

    cards   = []
    fig_no  = 0

    NSMAP = {
        "w":   "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
        "r":   "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "wp":  "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
        "a":   "http://schemas.openxmlformats.org/drawingml/2006/main",
        "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    }

    try:
        with zipfile.ZipFile(str(path)) as zf:
            rels_xml = zf.read("word/_rels/document.xml.rels")
            rels_tree = etree.fromstring(rels_xml)
            rel_map = {
                r.get("Id"): r.get("Target")
                for r in rels_tree
                if "image" in r.get("Target", "").lower()
            }

            doc_xml  = zf.read("word/document.xml")
            doc_tree = etree.fromstring(doc_xml)
            body     = doc_tree.find(
                ".//w:body",
                namespaces={"w": NSMAP["w"]}
            )

            paragraphs_before = []
            para_idx          = 0
            module_id         = "module_1"

            for elem in body:
                tag = etree.QName(elem.tag).localname

                if tag == "p":
                    texts = elem.findall(
                        ".//w:t",
                        namespaces={"w": NSMAP["w"]}
                    )
                    para_text = "".join(
                        t.text or "" for t in texts
                    ).strip()

                    if re.match(
                        r"(module|chapter|unit|section)\s+\d+",
                        para_text, re.I
                    ):
                        para_idx += 1
                        module_id = f"module_{para_idx}"

                    if para_text:
                        paragraphs_before.append(para_text)
                        if len(paragraphs_before) > 5:
                            paragraphs_before.pop(0)

                    blips = elem.findall(
                        ".//a:blip",
                        namespaces={"a": NSMAP["a"]}
                    )
                    for blip in blips:
                        r_embed = blip.get(
                            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
                        )
                        if r_embed and r_embed in rel_map:
                            img_target = rel_map[r_embed]
                            img_name   = f"word/{img_target.lstrip('/')}"

                            try:
                                img_bytes = zf.read(img_name)
                            except Exception:
                                continue

                            if len(img_bytes) < 1000:
                                continue

                            ext    = Path(img_name).suffix.lstrip(".")
                            fig_no += 1
                            fig_id = (
                                f"doc_{doc_id}_docx_f{fig_no:02d}"
                            )
                            filename = f"{fig_id}.{ext}"
                            img_path = asset_dir / filename
                            img_path.write_bytes(img_bytes)

                            preceding = " ".join(
                                paragraphs_before[-3:]
                            )

                            explicit = [
                                p for p in paragraphs_before
                                if _REF_RE.search(p)
                            ]

                            ocr_text = _quick_ocr(img_bytes)

                            combined = f"{ocr_text} {preceding}"
                            vtype    = _classify_visual_type(combined)

                            card = FigureCard(
                                id             = fig_id,
                                document_id    = doc_id,
                                module_id      = module_id,
                                page           = fig_no,
                                figure_index   = fig_no,
                                image_path     = str(img_path),
                                image_url      = (
                                    f"{image_url_prefix}/{filename}"
                                ),
                                caption        = "",
                                ocr_text       = ocr_text,
                                preceding_text = preceding,
                                following_text = "",
                                section_title  = "",
                                explicit_refs  = explicit,
                                visual_type    = vtype,
                            )
                            card.provenance_score = (
                                _compute_provenance_score(card)
                            )

                            if not ocr_text and not explicit:
                                card.eligible    = False
                                card.skip_reason = "no_evidence"

                            cards.append(card)

    except Exception as e:
        print(f"[FIGURE] DOCX extraction error: {e}")

    print(f"[FIGURE] DOCX: extracted {len(cards)} figures from {path.name}")
    return cards


# -- Public API ------------------------------------------------

def extract_figures(
    file_path:        str,
    doc_id:           str,
    module_map:       dict[int, str],
    asset_dir:        str  = "extracted_output/assets",
    image_url_prefix: str  = "/api/assets",
) -> list[FigureCard]:
    path      = Path(file_path)
    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()

    if ext == ".pdf":
        return _extract_from_pdf(
            path, doc_id, module_map,
            asset_dir, image_url_prefix
        )
    elif ext == ".docx":
        return _extract_from_docx(
            path, doc_id, module_map,
            asset_dir, image_url_prefix
        )
    else:
        print(f"[FIGURE] Unsupported type for figure extraction: {ext}")
        return []
