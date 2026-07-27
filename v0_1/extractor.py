"""
AION Module: Extractor
Maturity:    v0.1 — PYMUPDF / TEXT EXTRACTOR
Upgrades to: Document Intelligence Parser Engine (LayoutLMv3 / OCR / Table Extraction)
Contract:    source_path: str -> Document (see schemas.py)
"""

from pathlib import Path
import uuid
from .schemas import Document
from .content_filter import extract_academic_content, AcademicContentFilter
from .material_classifier import classify_material

def extract(pdf_or_text_path: str) -> Document:
    path = Path(pdf_or_text_path)
    file_type = path.suffix.lstrip(".").lower() or "txt"
    text = ""
    report = {}

    if file_type == "pdf":
        classification = classify_material(str(path))
        mat_type = classification.material_type  # keep lowercase: "textbook", "notes"
        print(f"[EXTRACTOR] Document Classified as '{mat_type}' (Confidence: {classification.confidence})", flush=True)
        text, report = extract_academic_content(str(path), material_type=mat_type)
        kept = len(report.get("kept_pages", []))
        total = report.get("total_pages", 0)
        print(f"[EXTRACTOR] Kept {report.get('kept_word_count', 0)} words across {kept}/{total} pages (Method: {report.get('method')})", flush=True)
        if total > 20 and kept < total * 0.15:
            print(f"[EXTRACTOR] ⚠ WARNING: Only {kept}/{total} pages kept. Content filter may be too aggressive.", flush=True)
    else:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        filt = AcademicContentFilter()
        pages = raw.split("\n\n")
        text, report_obj = filt.filter_text_pages(pages)

    doc_id = str(uuid.uuid4())[:8]

    # Save artifact copy in output folder
    import json
    output_dir = Path("extracted_output")
    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / f"{path.stem}_{doc_id}.txt"
    out_file.write_text(text, encoding="utf-8")
    if report:
        report_out = output_dir / "last_report.json"
        report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return Document(
        doc_id=doc_id,
        source_path=str(path),
        raw_text=text,
        file_type=file_type,
    )
