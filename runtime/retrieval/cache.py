# runtime/retrieval/cache.py
"""Versioned extraction and index cache for the LAPTOP_FAST profile.

Cache layout under `.aion_cache/`:
    manifest.json           — per-module source hashes + pipeline version tags
    extraction/M1.json      — cached extraction output for module 1
    extraction/M2.json      — ...
    bm25/module_1/          — serialised BM25 index for module 1
    bm25/module_2/          — ...

A cache entry is valid only if ALL of these match:
    source_sha256, extraction_version, chunking_version,
    validation_version, bm25_version
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

# Current pipeline component versions — bump these when the component changes
EXTRACTION_VERSION = "v5"
CHUNKING_VERSION = "v3"
VALIDATION_VERSION = "v5"
BM25_VERSION = "v1"

_CACHE_ROOT = Path(".aion_cache")


@dataclass
class ModuleCacheEntry:
    """Cache metadata for a single module."""

    module_id: str
    source_sha256: str
    extraction_version: str = EXTRACTION_VERSION
    chunking_version: str = CHUNKING_VERSION
    validation_version: str = VALIDATION_VERSION
    bm25_version: str = BM25_VERSION


@dataclass
class CacheManifest:
    """Top-level cache manifest tracking all modules."""

    modules: Dict[str, ModuleCacheEntry] = field(default_factory=dict)
    dataset_hash: str = ""


class ExtractionCache:
    """Read/write cache for extracted text and BM25 indexes."""

    def __init__(self, cache_root: Optional[Path] = None):
        self.root = (cache_root or _CACHE_ROOT).resolve()
        self.manifest_path = self.root / "manifest.json"
        self.extraction_dir = self.root / "extraction"
        self.bm25_dir = self.root / "bm25"
        self._manifest: Optional[CacheManifest] = None

    # -- Manifest I/O --------------------------------------------------

    def _load_manifest(self) -> CacheManifest:
        if self._manifest is not None:
            return self._manifest
        if not self.manifest_path.exists():
            self._manifest = CacheManifest()
            return self._manifest
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            modules = {}
            for mid, entry in raw.get("modules", {}).items():
                modules[mid] = ModuleCacheEntry(**entry)
            self._manifest = CacheManifest(
                modules=modules,
                dataset_hash=raw.get("dataset_hash", ""),
            )
        except (json.JSONDecodeError, OSError, TypeError):
            self._manifest = CacheManifest()
        return self._manifest

    def _save_manifest(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self._load_manifest()
        raw = {
            "dataset_hash": manifest.dataset_hash,
            "modules": {
                mid: asdict(entry) for mid, entry in manifest.modules.items()
            },
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2)

    # -- Cache Query ---------------------------------------------------

    def is_valid(self, module_id: str, source_sha256: str) -> bool:
        """Check if the cache for a module is still valid."""
        manifest = self._load_manifest()
        entry = manifest.modules.get(module_id)
        if entry is None:
            return False
        return (
            entry.source_sha256 == source_sha256
            and entry.extraction_version == EXTRACTION_VERSION
            and entry.chunking_version == CHUNKING_VERSION
            and entry.validation_version == VALIDATION_VERSION
            and entry.bm25_version == BM25_VERSION
        )

    # -- Extraction Cache ----------------------------------------------

    def get_extraction(self, module_id: str) -> Optional[Any]:
        """Load cached extraction output for a module."""
        path = self.extraction_dir / f"{module_id}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def save_extraction(
        self, module_id: str, source_sha256: str, data: Any
    ) -> None:
        """Save extraction output and update manifest."""
        self.extraction_dir.mkdir(parents=True, exist_ok=True)
        path = self.extraction_dir / f"{module_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        manifest = self._load_manifest()
        manifest.modules[module_id] = ModuleCacheEntry(
            module_id=module_id,
            source_sha256=source_sha256,
        )
        self._save_manifest()

    # -- BM25 Index Cache ----------------------------------------------

    def get_bm25_dir(self, module_id: str) -> Path:
        """Return the directory for a module's BM25 index files."""
        module_num = module_id.replace("M", "module_")
        return self.bm25_dir / module_num

    def bm25_exists(self, module_id: str) -> bool:
        """Check if a serialised BM25 index exists for a module."""
        d = self.get_bm25_dir(module_id)
        return d.exists() and any(d.iterdir())

    # -- Dataset Hash --------------------------------------------------

    def set_dataset_hash(self, combined_hash: str) -> None:
        manifest = self._load_manifest()
        manifest.dataset_hash = combined_hash
        self._save_manifest()

    def get_dataset_hash(self) -> str:
        return self._load_manifest().dataset_hash
