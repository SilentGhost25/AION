"""
AION Module: Extractor
Handles PDF, DOCX, TXT, MD files correctly.
"""

from pathlib import Path
import uuid
import json
from .schemas import Document
from .content_filter import extract_academic_content, AcademicContentFilter
from .material_classifier import classify_material

USE_NEW_PARSER = True


def extract(pdf_or_text_path: str) -> Document:
    path      = Path(pdf_or_text_path)
    file_type = path.suffix.lstrip(".").lower() or "txt"
    text      = ""
    report    = {}

    print(f"[EXTRACTOR] File: {path.name} | Type: {file_type}", flush=True)

    # ── DOCX ──────────────────────────────────────────────────
    if file_type == "docx":
        try:
            from .docx_parser import extract_docx_text
            text   = extract_docx_text(str(path))
            report = {
                "method":     "python-docx",
                "word_count": len(text.split()),
            }
        except ImportError:
            raise RuntimeError(
                "python-docx not installed. Run: pip install python-docx"
            )
        except Exception as e:
            raise RuntimeError(f"DOCX extraction failed: {e}")

    # ── PDF ───────────────────────────────────────────────────
    elif file_type == "pdf":
        if USE_NEW_PARSER:
            try:
                from .document_parser import parse_document
                parsed = parse_document(str(path))
                text   = parsed.full_text_with_tables()
                report = {
                    "method":       parsed.method,
                    "word_count":   parsed.word_count,
                    "confidence":   parsed.confidence,
                    "tables_found": len(parsed.tables),
                    "ocr_used":     parsed.ocr_used,
                }
            except Exception as err:
                print(f"[EXTRACTOR] New parser failed: {err} — trying fallback", flush=True)
                try:
                    classification = classify_material(str(path))
                    text, report   = extract_academic_content(
                        str(path),
                        material_type=classification.material_type
                    )
                except Exception as e2:
                    raise RuntimeError(f"PDF extraction failed: {err} | {e2}")
        else:
            classification = classify_material(str(path))
            text, report   = extract_academic_content(
                str(path),
                material_type=classification.material_type
            )

    # ── TXT / MD ──────────────────────────────────────────────
    elif file_type in ("txt", "md"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            report = {
                "method":     "direct-read",
                "word_count": len(text.split()),
            }
        except Exception as e:
            raise RuntimeError(f"Text extraction failed: {e}")

    else:
        raise RuntimeError(
            f"Unsupported file type: .{file_type}. Supported: pdf, docx, txt, md"
        )

    # ── Validate ──────────────────────────────────────────────
    word_count = len(text.split()) if text else 0
    print(f"[EXTRACTOR] Extracted {word_count} words", flush=True)

    if word_count < 10:
        raise RuntimeError(
            f"Extraction failed — only {word_count} words extracted from {path.name}. File may be corrupted or empty."
        )

    # ── Save artifact ─────────────────────────────────────────
    doc_id     = str(uuid.uuid4())[:8]
    output_dir = Path("extracted_output")
    output_dir.mkdir(exist_ok=True)

    out_file = output_dir / f"{path.stem}_{doc_id}.txt"
    out_file.write_text(text, encoding="utf-8")

    if report:
        (output_dir / "last_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    print(f"[EXTRACTOR] Saved to {out_file}", flush=True)

    return Document(
        doc_id      = doc_id,
        source_path = str(path),
        raw_text    = text,
        file_type   = file_type,
    )
