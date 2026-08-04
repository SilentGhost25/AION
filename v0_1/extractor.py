"""
AION Module: Extractor
Enhanced: ConfidenceGatedExtractor with formula extraction and OCR fallback validation.
"""

from pathlib import Path
import uuid
import re
from typing import List, Dict, Any
from .schemas import Document
from .content_filter import extract_academic_content, AcademicContentFilter
from .material_classifier import classify_material
from .formula_extractor import extract_formulas

USE_NEW_PARSER = True


class ConfidenceGatedExtractor:
    """
    Applies extraction strategies based on text confidence:
    >=80% : Native PyMuPDF
    60-80%: PyMuPDF + OCR page validation
    <60%  : Full OCR override
    """

    THRESHOLDS = {
        "high":   0.80,
        "medium": 0.60,
        "low":    0.40,
    }

    def extract_pdf(self, pdf_path: str) -> Dict[str, Any]:
        from .document_parser import parse_document
        parsed = parse_document(pdf_path)
        text = parsed.full_text_with_tables()
        confidence = getattr(parsed, "confidence", 0.58)

        print(f"[EXTRACTOR] Primary confidence: {confidence:.0%}")

        if confidence >= self.THRESHOLDS["high"]:
            print(f"[EXTRACTOR] Strategy: native text (confidence={confidence:.0%})")
            return {"text": text, "method": parsed.method, "confidence": confidence}

        if confidence >= self.THRESHOLDS["medium"]:
            print(f"[EXTRACTOR] Strategy: OCR validation (confidence={confidence:.0%})")
            text = self._validate_with_ocr(pdf_path, text)
            return {"text": text, "method": "pymupdf+ocr_validated", "confidence": 0.75}

        print(f"[EXTRACTOR] Strategy: OCR override (confidence={confidence:.0%})")
        ocr_text = self._extract_full_ocr(pdf_path)
        if ocr_text.strip():
            return {"text": ocr_text, "method": "rapidocr_full", "confidence": 0.72}

        return {"text": text, "method": parsed.method, "confidence": confidence}

    def _validate_with_ocr(self, pdf_path: str, primary_text: str) -> str:
        try:
            from rapidocr import RapidOCR
            import fitz
            ocr = RapidOCR()
            doc = fitz.open(pdf_path)
            enhanced_pages = []

            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if len(page_text.strip().split()) < 30:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_bytes = pix.tobytes("png")
                    ocr_res, _ = ocr(img_bytes)
                    if ocr_res:
                        lines = [r[1] for r in ocr_res if r[2] > 0.5]
                        page_text = " ".join(lines)
                enhanced_pages.append(page_text)

            return "\n\n".join(enhanced_pages)
        except Exception as e:
            print(f"[EXTRACTOR] OCR validation fallback failed: {e}")
            return primary_text

    def _extract_full_ocr(self, pdf_path: str) -> str:
        try:
            from rapidocr import RapidOCR
            import fitz
            ocr = RapidOCR()
            doc = fitz.open(pdf_path)
            blocks = []

            for page_num, page in enumerate(doc):
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_bytes = pix.tobytes("png")
                result, _ = ocr(img_bytes)
                if result:
                    lines = [r[1] for r in result if r[2] > 0.50]
                    t = " ".join(lines)
                    if t.strip():
                        blocks.append(t)

            return "\n\n".join(blocks)
        except Exception as e:
            print(f"[EXTRACTOR] Full OCR failed: {e}")
            return ""


def _clean_extracted_text(text: str) -> str:
    """Remove excess blank lines while preserving structure and math formulas."""
    text = re.sub(r"\n{4,}", "\n\n", text)
    text = re.sub(r"[ \t]{3,}", " ", text)
    return text.strip()


def extract(pdf_or_text_path: str) -> Document:
    path      = Path(pdf_or_text_path)
    file_type = path.suffix.lstrip(".").lower() or "txt"
    text      = ""
    report    = {}

    print(f"[EXTRACTOR] File: {path.name} | Type: {file_type}", flush=True)

    if file_type in ("txt", "md"):
        raw   = path.read_text(encoding="utf-8", errors="ignore")
        filt  = AcademicContentFilter()
        pages = raw.split("\n\n")
        text, _ = filt.filter_text_pages(pages)
        report = {"method": "text_filter", "word_count": len(text.split())}

    elif file_type == "docx":
        try:
            from .docx_parser import extract_docx_text
            text   = extract_docx_text(str(path))
            report = {"method": "python-docx", "word_count": len(text.split())}
        except Exception as e:
            raise RuntimeError(f"DOCX extraction failed: {e}")

    else:
        try:
            gated = ConfidenceGatedExtractor()
            res = gated.extract_pdf(str(path))
            text = res["text"]
            report = {
                "method":     res["method"],
                "word_count": len(text.split()),
                "confidence": res["confidence"],
            }
        except Exception as e:
            raise RuntimeError(f"PDF extraction failed: {e}")

    text = _clean_extracted_text(text)
    word_count = len(text.split()) if text else 0
    if word_count < 20:
        raise RuntimeError(
            f"Extraction failed — {word_count} words. File may be images-only."
        )

    extracted_formula_objs = extract_formulas(text)
    formulas = [f.raw for f in extracted_formula_objs[:10]]
    if formulas:
        report["formulas"] = formulas

    doc_id     = str(uuid.uuid4())[:8]
    output_dir = Path("extracted_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{path.stem}_{doc_id}.txt"
    out_file.write_text(text, encoding="utf-8")

    return Document(
        doc_id      = doc_id,
        source_path = str(path),
        raw_text    = text,
        file_type   = file_type,
        formulas    = formulas,
    )
