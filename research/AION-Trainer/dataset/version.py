"""
Dataset Versioning

Instead of overwriting, every dataset build creates a new version:
    dataset/
        BAI401/
            v1/
            v2/
            v3/
            latest -> v3
"""

import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger("aion.dataset.version")


class DatasetVersion:
    """Manages dataset versions."""

    def __init__(self, base_dir: str = "dataset/"):
        self.base_dir = Path(base_dir)

    def get_latest_version(self, subject_code: str) -> Optional[str]:
        """Get the latest version for a subject."""
        subject_dir = self.base_dir / subject_code
        if not subject_dir.exists():
            return None

        versions = []
        for d in subject_dir.iterdir():
            if d.is_dir() and d.name.startswith("v"):
                try:
                    versions.append(int(d.name[1:]))
                except ValueError:
                    pass

        return f"v{max(versions)}" if versions else None

    def get_next_version(self, subject_code: str) -> str:
        """Get the next version number."""
        latest = self.get_latest_version(subject_code)
        if latest:
            num = int(latest[1:])
            return f"v{num + 1}"
        return "v1"

    def create_version(
        self,
        subject_code: str,
        source_files: List[str],
        num_objects: int,
    ) -> Dict[str, Any]:
        """Create a new dataset version."""
        version = self.get_next_version(subject_code)
        version_dir = self.base_dir / subject_code / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy current dataset files
        subject_dir = self.base_dir / subject_code
        for f in subject_dir.glob("*.jsonl"):
            shutil.copy2(f, version_dir / f.name)
        for f in subject_dir.glob("metadata.json"):
            shutil.copy2(f, version_dir / f.name)

        # Create version info
        version_info = {
            "version": version,
            "subject_code": subject_code,
            "created_at": datetime.utcnow().isoformat(),
            "source_files": source_files,
            "num_objects": num_objects,
        }

        version_info_file = version_dir / "version_info.json"
        with open(version_info_file, "w", encoding="utf-8") as f:
            json.dump(version_info, f, indent=2)

        # Update latest symlink
        latest_link = subject_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        try:
            latest_link.symlink_to(version_dir, target_is_directory=True)
        except OSError:
            # Fallback for Windows environments where symlink creation is restricted
            with open(latest_link, "w", encoding="utf-8") as f:
                f.write(version_dir.name)

        logger.info(f"Created dataset version: {version}")
        return version_info

    def list_versions(self, subject_code: str) -> List[Dict[str, Any]]:
        """List all versions for a subject."""
        subject_dir = self.base_dir / subject_code
        if not subject_dir.exists():
            return []

        versions = []
        for d in sorted(subject_dir.iterdir()):
            if d.is_dir() and d.name.startswith("v"):
                info_file = d / "version_info.json"
                if info_file.exists():
                    with open(info_file, encoding="utf-8") as f:
                        versions.append(json.load(f))

        return versions
