import os
import json
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import asdict
from .document_model import AcademicDocument, Section

class CacheManager:
    """
    Manages caching of extracted documents to avoid re-running expensive intelligence extraction.
    Uses SHA256 of the source file to identify unique documents.
    """
    def __init__(self, workspace_dir: str = "workspace"):
        self.workspace_dir = Path(workspace_dir)
        self.documents_dir = self.workspace_dir / "documents"
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    def compute_sha256(self, file_path: str) -> str:
        """Computes the SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get_cache_dir(self, file_path: str) -> Path:
        """Returns the cache directory for a given file based on its hash."""
        file_hash = self.compute_sha256(file_path)
        return self.documents_dir / file_hash

    def load_cached_document(self, file_path: str) -> Optional[AcademicDocument]:
        """Attempts to load a cached AcademicDocument for the given file."""
        cache_dir = self.get_cache_dir(file_path)
        metadata_path = cache_dir / "metadata.json"
        
        if not metadata_path.exists():
            return None
            
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            doc = AcademicDocument(source_path=file_path)
            doc.metadata = data.get("metadata", {})
            doc.json_structure = data.get("json_structure", {})
            raw_sections = data.get("sections", [])
            doc.sections = [
                Section(
                    title=s.get("title", ""),
                    content=s.get("content", ""),
                    layout_type=s.get("layout_type", ""),
                    page_range=tuple(s["page_range"]) if s.get("page_range") else None,
                )
                if isinstance(s, dict) else s
                for s in raw_sections
            ]
            doc.tables = data.get("tables", [])
            doc.figures = data.get("figures", [])
            doc.formulas = data.get("formulas", [])
            doc.toc = data.get("toc", [])
            doc.reading_order = data.get("reading_order", [])
            doc.page_map = data.get("page_map", {})
            
            markdown_path = cache_dir / "markdown.md"
            if markdown_path.exists():
                with open(markdown_path, 'r', encoding='utf-8') as f:
                    doc.markdown = f.read()
                    
            return doc
        except Exception as e:
            print(f"Error loading cached document: {e}")
            return None

    def save_document(self, doc: AcademicDocument):
        """Saves an AcademicDocument to the cache."""
        cache_dir = self.get_cache_dir(doc.source_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save markdown
        if doc.markdown:
            with open(cache_dir / "markdown.md", 'w', encoding='utf-8') as f:
                f.write(doc.markdown)
                
        # Save structured data
        data = {
            "metadata": doc.metadata,
            "json_structure": doc.json_structure,
            "sections": [asdict(s) if isinstance(s, Section) else s for s in doc.sections],
            "tables": doc.tables,
            "figures": doc.figures,
            "formulas": doc.formulas,
            "toc": doc.toc,
            "reading_order": doc.reading_order,
            "page_map": doc.page_map
        }
        
        with open(cache_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
