"""
AION v2 Content-Addressable Artifact Registry
============================================
Canonical, portable repository for all document artifacts and binary assets.
Resolves asset keys dynamically to local storage paths, ensuring artifacts
remain decoupled from specific machine file paths.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from aion.core.dom.artifacts import (
    ArtifactType,
    BaseArtifact,
    FigureArtifact,
    create_artifact_from_dict,
)


class ArtifactRegistry:
    """
    Central registry managing in-memory artifacts and portable asset resolution.
    """

    def __init__(self, assets_root_dir: Optional[Union[str, Path]] = None):
        self._artifacts: Dict[str, BaseArtifact] = {}
        self._by_module: Dict[int, List[str]] = {}
        self._by_type: Dict[ArtifactType, List[str]] = {}
        self._by_section: Dict[str, List[str]] = {}
        self._by_page: Dict[int, List[str]] = {}

        # Default asset root directory: workspace/artifacts
        if assets_root_dir is None:
            base = Path(os.environ.get("AION_BASE_DIR") or Path.cwd())
            self.assets_root_dir = base / "workspace" / "artifacts"
        else:
            self.assets_root_dir = Path(assets_root_dir)

    def register(self, artifact: BaseArtifact) -> None:
        """Register an artifact and update all secondary indices."""
        aid = artifact.artifact_id
        self._artifacts[aid] = artifact

        # Module index
        self._by_module.setdefault(artifact.module_index, []).append(aid)

        # Type index
        self._by_type.setdefault(artifact.artifact_type, []).append(aid)

        # Section index
        if artifact.section_id:
            self._by_section.setdefault(artifact.section_id, []).append(aid)

        # Page index
        self._by_page.setdefault(artifact.page_number, []).append(aid)

    def get(self, artifact_id: str) -> Optional[BaseArtifact]:
        """Retrieve an artifact by its canonical ID."""
        return self._artifacts.get(artifact_id)

    def contains(self, artifact_id: str) -> bool:
        """Check if an artifact exists in the registry."""
        return artifact_id in self._artifacts

    def count(self) -> int:
        """Return total registered artifacts count."""
        return len(self._artifacts)

    def all(self) -> List[BaseArtifact]:
        """Retrieve all registered artifacts in insertion order."""
        return list(self._artifacts.values())

    def __iter__(self):
        return iter(self._artifacts.values())

    def get_by_module(self, module_index: int) -> List[BaseArtifact]:
        """Retrieve all artifacts belonging to a specific module."""
        aids = self._by_module.get(module_index, [])
        return [self._artifacts[aid] for aid in aids if aid in self._artifacts]

    def get_by_type(self, artifact_type: ArtifactType) -> List[BaseArtifact]:
        """Retrieve all artifacts of a specific type."""
        aids = self._by_type.get(artifact_type, [])
        return [self._artifacts[aid] for aid in aids if aid in self._artifacts]

    def get_by_section(self, section_id: str) -> List[BaseArtifact]:
        """Retrieve all artifacts belonging to a specific section."""
        aids = self._by_section.get(section_id, [])
        return [self._artifacts[aid] for aid in aids if aid in self._artifacts]

    def get_by_page(self, page_number: int) -> List[BaseArtifact]:
        """Retrieve all artifacts appearing on a specific page."""
        aids = self._by_page.get(page_number, [])
        return [self._artifacts[aid] for aid in aids if aid in self._artifacts]

    # -----------------------------------------------------------------------
    # Portable Asset Resolution
    # -----------------------------------------------------------------------

    def resolve_asset_path(self, asset_key_or_artifact: Union[FigureArtifact, str]) -> Path:
        """
        Dynamically map an abstract asset key to a local filesystem Path.
        Never stores absolute machine paths in the artifact itself.
        """
        if isinstance(asset_key_or_artifact, FigureArtifact):
            key = asset_key_or_artifact.asset_key
        else:
            key = str(asset_key_or_artifact)

        clean_key = key.lstrip("/\\")
        return (self.assets_root_dir / clean_key).resolve()

    def resolve_figure_path(self, figure_id: str) -> Optional[Path]:
        """
        Look up a figure by its canonical figure_id and resolve its local file path.
        """
        art = self.get(figure_id)
        if not isinstance(art, FigureArtifact) or not art.asset_key:
            return None
        return self.resolve_asset_path(art.asset_key)

    def store_asset_bytes(
        self,
        image_bytes: bytes,
        relative_subpath: str = "figures",
        filename_hint: Optional[str] = None
    ) -> Tuple[str, str, Path]:
        """
        Content-addressable storage for binary assets (PNG/SVG/WebP).
        Saves bytes to disk using its SHA256 digest to deduplicate repeating assets.
        Returns: (asset_key, sha256_hash, local_path)
        """
        sha = hashlib.sha256(image_bytes).hexdigest()
        ext = ".png"
        if filename_hint and "." in filename_hint:
            ext = os.path.splitext(filename_hint)[1].lower()

        filename = f"{sha[:16]}{ext}"
        asset_key = f"{relative_subpath.strip('/\\')}/{filename}"
        local_path = self.resolve_asset_path(asset_key)

        local_path.parent.mkdir(parents=True, exist_ok=True)
        if not local_path.exists():
            with open(local_path, "wb") as f:
                f.write(image_bytes)

        return asset_key, sha, local_path

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all artifacts to dictionary. No absolute paths included."""
        return {
            "assets_root_dir": str(self.assets_root_dir),
            "artifacts": {aid: a.to_dict() for aid, a in self._artifacts.items()},
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], assets_root_dir: Optional[Union[str, Path]] = None
    ) -> ArtifactRegistry:
        root = assets_root_dir or data.get("assets_root_dir")
        reg = cls(assets_root_dir=root)
        raw_map = data.get("artifacts", {})
        for aid, a_data in raw_map.items():
            reg.register(create_artifact_from_dict(a_data))
        return reg
