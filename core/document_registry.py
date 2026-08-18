"""
AION Document Registry
======================
Single source of truth for all uploaded documents.
Replaces the raw file_registry dict in aion_api.py.

Every document goes through states:
  UPLOADED -> EXTRACTING -> EXTRACTED -> INDEXING -> READY -> GENERATING -> DONE

Nothing touches the filesystem path after READY.
"""

import json
import threading
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class DocumentStatus(str, Enum):
    UPLOADED   = "uploaded"
    EXTRACTING = "extracting"
    EXTRACTED  = "extracted"
    INDEXING   = "indexing"
    READY      = "ready"
    GENERATING = "generating"
    FAILED     = "failed"


class Document:
    """Represents a single uploaded document through its full lifecycle."""

    def __init__(
        self,
        doc_id:    str,
        filename:  str,
        path:      str,
        subject:   str = "unknown",
        category:  str = "notes",
    ):
        self.id            = doc_id
        self.filename      = filename
        self.path          = path          # only used during extraction
        self.subject       = subject
        self.category      = category
        self.status        = DocumentStatus.UPLOADED
        self.uploaded_at   = datetime.now().isoformat()
        self.size_bytes    = Path(path).stat().st_size if Path(path).exists() else 0

        # Populated after extraction
        self.modules:      list[dict] = []
        self.chunks:       list[dict] = []
        self.figures:      list[dict] = []
        self.word_count:   int        = 0
        self.confidence:   float      = 0.0
        self.error:        str        = ""

        # Cache paths
        self._cache_dir: Optional[Path] = None

    def set_cache_dir(self, base: Path):
        self._cache_dir = base / self.id
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def save_modules(self, modules: list[dict]):
        self.modules = modules
        if self._cache_dir:
            with open(self._cache_dir / "modules.json", "w") as f:
                json.dump(modules, f, indent=2)

    def save_chunks(self, chunks: list[dict]):
        self.chunks = chunks
        if self._cache_dir:
            with open(self._cache_dir / "chunks.json", "w") as f:
                json.dump(chunks, f, indent=2)

    def save_figures(self, figures: list[dict]):
        self.figures = figures
        if self._cache_dir:
            with open(self._cache_dir / "figures.json", "w") as f:
                json.dump(figures, f, indent=2)

    def load_cache(self) -> bool:
        """Load from disk cache if available. Returns True if cache hit."""
        if not self._cache_dir:
            return False
        try:
            m = self._cache_dir / "modules.json"
            c = self._cache_dir / "chunks.json"
            if m.exists() and c.exists():
                with open(m) as f:
                    self.modules = json.load(f)
                with open(c) as f:
                    self.chunks = json.load(f)
                return True
        except Exception:
            pass
        return False

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "filename":    self.filename,
            "subject":     self.subject,
            "category":    self.category,
            "status":      self.status.value,
            "uploaded_at": self.uploaded_at,
            "size_bytes":  self.size_bytes,
            "word_count":  self.word_count,
            "confidence":  self.confidence,
            "module_count": len(self.modules),
            "chunk_count":  len(self.chunks),
            "figure_count": len(self.figures),
            "error":        self.error,
        }


class DocumentRegistry:
    """
    Thread-safe registry of all documents.
    Replaces file_registry dict and job_store dict.
    """

    def __init__(self, cache_dir: Path):
        self._docs:  dict[str, Document] = {}
        self._lock   = threading.Lock()
        self.cache   = cache_dir
        self.cache.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        filename: str,
        path:     str,
        subject:  str = "unknown",
        category: str = "notes",
    ) -> Document:
        doc_id = str(uuid.uuid4())[:12]
        doc    = Document(doc_id, filename, path, subject, category)
        doc.set_cache_dir(self.cache)

        with self._lock:
            self._docs[doc_id] = doc

        return doc

    def get(self, doc_id: str) -> Optional[Document]:
        return self._docs.get(doc_id)

    def set_status(self, doc_id: str, status: DocumentStatus, error: str = ""):
        doc = self.get(doc_id)
        if doc:
            doc.status = status
            if error:
                doc.error = error

    def all(self) -> list[dict]:
        with self._lock:
            return [d.to_dict() for d in self._docs.values()]

    def exists(self, doc_id: str) -> bool:
        return doc_id in self._docs
