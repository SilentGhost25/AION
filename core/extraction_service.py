"""
AION Extraction Service
=======================
Runs in a background thread after upload.
Populates the Document with modules and chunks.
Generation never re-extracts.
"""

import threading
import traceback
from pathlib import Path
from typing import Callable, Optional

from core.document_registry import DocumentRegistry, DocumentStatus


class ExtractionService:
    """
    Runs document extraction in the background.
    Updates document status at each stage.
    """

    def __init__(self, registry: DocumentRegistry):
        self.registry = registry

    def extract_async(
        self,
        doc_id:     str,
        on_ready:   Optional[Callable] = None,
        on_error:   Optional[Callable] = None,
    ):
        """Start extraction in a background thread."""
        thread = threading.Thread(
            target  = self._run,
            args    = (doc_id, on_ready, on_error),
            daemon  = True,
            name    = f"extract-{doc_id}",
        )
        thread.start()
        return thread

    def _run(self, doc_id: str, on_ready, on_error):
        doc = self.registry.get(doc_id)
        if not doc:
            return

        try:
            # ── Stage 1: Extracting ───────────────────────────────────────────
            self.registry.set_status(doc_id, DocumentStatus.EXTRACTING)
            print(f"[EXTRACT] Starting: {doc.filename}")

            path = Path(doc.path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {doc.path}")

            # Use existing AION extraction pipeline
            try:
                from v0_1.extractor import Extractor
                extractor = Extractor()
                extraction = extractor.extract(str(path))
                text       = extraction.get("text", "")
                confidence = extraction.get("confidence", 0.0)
            except Exception as e:
                print(f"[EXTRACT] Extractor failed: {e} — using fallback")
                text, confidence = self._fallback_extract(path)

            doc.word_count = len(text.split())
            doc.confidence = confidence
            print(f"[EXTRACT] Done: {doc.word_count} words, conf={confidence:.0%}")

            # ── Stage 2: Module detection ─────────────────────────────────────
            self.registry.set_status(doc_id, DocumentStatus.EXTRACTED)
            print(f"[SEGMENT] Detecting modules...")

            try:
                from v0_1.segmenter import Segmenter
                segmenter = Segmenter()
                segments  = segmenter.segment(text)
                modules   = self._build_modules(segments)
                chunks    = self._build_chunks(segments)
            except Exception as e:
                print(f"[SEGMENT] Segmenter failed: {e} — using fallback")
                modules, chunks = self._fallback_segment(text, doc.filename)

            doc.save_modules(modules)
            doc.save_chunks(chunks)
            print(f"[SEGMENT] {len(modules)} modules, {len(chunks)} chunks")

            # ── Stage 3: Ready ────────────────────────────────────────────────
            self.registry.set_status(doc_id, DocumentStatus.READY)
            print(f"[EXTRACT] Document {doc_id} READY for generation")

            if on_ready:
                on_ready(doc_id)

        except Exception as e:
            err = str(e)
            print(f"[EXTRACT] FAILED {doc_id}: {err}")
            traceback.print_exc()
            self.registry.set_status(doc_id, DocumentStatus.FAILED, error=err)
            if on_error:
                on_error(doc_id, err)

    def _build_modules(self, segments: list) -> list[dict]:
        modules = []
        for i, seg in enumerate(segments):
            module_id = f"module_{i+1}"
            modules.append({
                "id":         module_id,
                "number":     i + 1,
                "title":      seg.get("title", f"Module {i+1}"),
                "word_count": seg.get("word_count", 0),
                "confidence": seg.get("confidence", 0.8),
                "pages":      seg.get("pages", []),
            })
        return modules

    def _build_chunks(self, segments: list) -> list[dict]:
        chunks = []
        chunk_id = 0
        for i, seg in enumerate(segments):
            module_id = f"module_{i+1}"
            text      = seg.get("text", "")
            # Split segment into ~300 word chunks
            words     = text.split()
            size      = 300
            for j in range(0, len(words), size):
                chunk_text = " ".join(words[j:j+size])
                if len(chunk_text.strip()) < 50:
                    continue
                chunks.append({
                    "id":        f"chunk_{chunk_id:04d}",
                    "module_id": module_id,
                    "module":    i + 1,
                    "text":      chunk_text,
                    "word_count": len(chunk_text.split()),
                })
                chunk_id += 1
        return chunks

    def _fallback_extract(self, path: Path) -> tuple[str, float]:
        """Simple text extraction when main extractor fails."""
        ext = path.suffix.lower()
        try:
            if ext == ".txt":
                text = path.read_text(encoding="utf-8", errors="ignore")
                return text, 0.85
            elif ext == ".pdf":
                import fitz
                doc  = fitz.open(str(path))
                text = "\n".join(page.get_text() for page in doc)
                doc.close()
                return text, 0.75
            elif ext in (".docx", ".doc"):
                import docx
                d    = docx.Document(str(path))
                text = "\n".join(p.text for p in d.paragraphs if p.text.strip())
                return text, 0.80
        except Exception as e:
            print(f"[EXTRACT] Fallback failed: {e}")
        return "", 0.0

    def _fallback_segment(
        self, text: str, filename: str
    ) -> tuple[list[dict], list[dict]]:
        """Simple segmentation when main segmenter fails."""
        words    = text.split()
        total    = len(words)
        n_mods   = min(5, max(1, total // 500))
        size     = total // n_mods if n_mods > 0 else total

        modules = []
        chunks  = []
        chunk_id = 0

        for i in range(n_mods):
            start     = i * size
            end       = start + size if i < n_mods - 1 else total
            mod_text  = " ".join(words[start:end])
            module_id = f"module_{i+1}"

            modules.append({
                "id":         module_id,
                "number":     i + 1,
                "title":      f"Module {i+1}",
                "word_count": end - start,
                "confidence": 0.7,
                "pages":      [],
            })

            # Chunk the module text
            mod_words = mod_text.split()
            for j in range(0, len(mod_words), 300):
                chunk_text = " ".join(mod_words[j:j+300])
                if len(chunk_text.strip()) < 50:
                    continue
                chunks.append({
                    "id":        f"chunk_{chunk_id:04d}",
                    "module_id": module_id,
                    "module":    i + 1,
                    "text":      chunk_text,
                    "word_count": len(chunk_text.split()),
                })
                chunk_id += 1

        return modules, chunks
