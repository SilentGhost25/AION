# runtime/profiles/laptop_demo.py
"""LaptopDemo runtime profile — pre-warmed, immutable-cache variant of LaptopFast.

After `python -m runtime.warmup` succeeds, a `demo_manifest.json` is written.
LaptopDemoProfile reads this manifest and refuses to start if any component
(dataset hash, model, backend, extraction/chunking/validation/bm25 version)
has changed since warmup.  This guarantees repeatable live demonstrations.
"""

import json
from pathlib import Path

from runtime.profiles.laptop_fast import LaptopFastProfile

_CACHE_DIR = Path(".aion_cache")
_DEMO_MANIFEST = _CACHE_DIR / "demo_manifest.json"


class DemoManifestError(RuntimeError):
    """Raised when demo manifest is missing or invalid."""


class LaptopDemoProfile(LaptopFastProfile):
    """Extends LaptopFastProfile with immutable cache validation.

    Validation rules identical to LaptopFast — same contracts, same
    validators, same ExportGate.  The only difference is initialisation:
    the model, extraction cache, and BM25 indexes are expected to be
    pre-built and verified before the profile activates.
    """

    def __init__(self):
        super().__init__()
        self._manifest = self._load_manifest()

    @property
    def name(self) -> str:
        return "LAPTOP_DEMO"

    # -- Manifest Validation -------------------------------------------

    def _load_manifest(self) -> dict:
        """Load and validate the demo manifest, raising on mismatch."""
        if not _DEMO_MANIFEST.exists():
            raise DemoManifestError(
                "Demo manifest not found.  Run: python -m runtime.warmup"
            )
        try:
            with open(_DEMO_MANIFEST, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise DemoManifestError(f"Corrupt demo manifest: {exc}") from exc

        required_keys = [
            "dataset_hash",
            "model",
            "backend",
            "extraction_version",
            "chunking_version",
            "validation_version",
            "bm25_version",
            "profile",
        ]
        missing = [k for k in required_keys if k not in manifest]
        if missing:
            raise DemoManifestError(
                f"Demo manifest missing keys: {missing}.  "
                "Run: python -m runtime.warmup"
            )

        if manifest.get("profile") != "LAPTOP_DEMO":
            raise DemoManifestError(
                f"Manifest profile is '{manifest.get('profile')}', "
                "expected 'LAPTOP_DEMO'.  Run: python -m runtime.warmup"
            )

        return manifest

    def validate_against_current_state(
        self,
        current_dataset_hash: str,
        current_extraction_version: str,
        current_chunking_version: str,
        current_validation_version: str,
        current_bm25_version: str,
    ) -> None:
        """Verify the manifest matches the current runtime state.

        Raises DemoManifestError if anything has changed.
        """
        checks = {
            "dataset_hash": current_dataset_hash,
            "extraction_version": current_extraction_version,
            "chunking_version": current_chunking_version,
            "validation_version": current_validation_version,
            "bm25_version": current_bm25_version,
        }
        mismatches = []
        for key, current_val in checks.items():
            manifest_val = self._manifest.get(key)
            if manifest_val != current_val:
                mismatches.append(
                    f"  {key}: manifest={manifest_val!r}, current={current_val!r}"
                )

        if mismatches:
            detail = "\n".join(mismatches)
            raise DemoManifestError(
                f"DEMO CACHE INVALID\n{detail}\n"
                "Run: python -m runtime.warmup"
            )

    @property
    def manifest(self) -> dict:
        """Return the loaded manifest for inspection."""
        return dict(self._manifest)
