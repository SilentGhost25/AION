# runtime/laptop_dataset.py
"""Scan the IAI dataset directory, identify module PDFs, and compute SHA-256 hashes.

Supports filename variations such as:
  - IAI-MODULE-1-NOTES.pdf
  - IAI-MODULE-3 NOTES.pdf   (space instead of hyphen)
"""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModuleFile:
    """Represents a single module's PDF file."""

    module_id: str  # e.g. "M1"
    module_number: int
    path: Path
    sha256: str
    size_bytes: int


# Regex matching IAI module note filenames with flexible separators
_MODULE_PATTERN = re.compile(
    r"IAI[-\s]MODULE[-\s](\d+)[-\s]NOTES\.pdf",
    re.IGNORECASE,
)


def _compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file in 64 KB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def scan_dataset(
    dataset_dir: str | Path,
    expected_modules: int = 5,
) -> Dict[str, ModuleFile]:
    """Scan the dataset directory for module PDFs.

    Returns a dict mapping module IDs ("M1" .. "M5") to ModuleFile objects.
    Raises ValueError if expected modules are missing.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    modules: Dict[str, ModuleFile] = {}

    for entry in dataset_path.iterdir():
        if not entry.is_file():
            continue
        match = _MODULE_PATTERN.match(entry.name)
        if not match:
            continue

        module_num = int(match.group(1))
        module_id = f"M{module_num}"

        if module_id in modules:
            raise ValueError(
                f"Duplicate module detected: {module_id} — "
                f"{modules[module_id].path} and {entry}"
            )

        modules[module_id] = ModuleFile(
            module_id=module_id,
            module_number=module_num,
            path=entry.resolve(),
            sha256=_compute_sha256(entry),
            size_bytes=entry.stat().st_size,
        )

    # Validate completeness
    missing = []
    for i in range(1, expected_modules + 1):
        mid = f"M{i}"
        if mid not in modules:
            missing.append(mid)

    if missing:
        raise ValueError(
            f"Missing module PDFs: {missing} in {dataset_path}.  "
            f"Found: {sorted(modules.keys())}"
        )

    return modules


def get_dataset_hash(modules: Dict[str, ModuleFile]) -> str:
    """Compute a combined hash representing the entire dataset state.

    This is used by the cache system to detect dataset changes.
    """
    h = hashlib.sha256()
    for mid in sorted(modules.keys()):
        h.update(mid.encode())
        h.update(modules[mid].sha256.encode())
    return h.hexdigest()


if __name__ == "__main__":
    import sys

    dataset = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Tarun J\Downloads\New Dataset\IAI"
    try:
        mods = scan_dataset(dataset)
        combined = get_dataset_hash(mods)
        print("=== IAI Dataset Scan ===")
        for mid in sorted(mods.keys()):
            m = mods[mid]
            print(f"  {m.module_id}: {m.path.name}  ({m.size_bytes:,} bytes)  sha256={m.sha256[:16]}…")
        print(f"\n  Combined hash: {combined[:24]}…")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
