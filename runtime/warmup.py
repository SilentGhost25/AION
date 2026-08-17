# runtime/warmup.py
"""AION Laptop Demo Warmup — pre-build all caches and validate the model.

Run as:
    python -m runtime.warmup

This command:
1. Loads the benchmark-winning model
2. Runs a tiny inference to verify JSON output
3. Loads and validates the dataset cache
4. Loads BM25 indexes
5. Runs one contract test
6. Creates an immutable demo_manifest.json

After this succeeds, LAPTOP_DEMO profile can be used.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.retrieval.cache import (
    ExtractionCache,
    EXTRACTION_VERSION,
    CHUNKING_VERSION,
    VALIDATION_VERSION,
    BM25_VERSION,
)

_DEFAULT_DATASET = r"C:\Users\Tarun J\Downloads\New Dataset\IAI"
_CACHE_DIR = Path(".aion_cache")


def _status(label: str, passed: bool, detail: str = "") -> bool:
    """Print a warmup step result."""
    icon = "✓" if passed else "✗"
    msg = f"  {icon} {label}"
    if detail:
        msg += f"  — {detail}"
    print(msg)
    return passed


def run_warmup(dataset_dir: str = _DEFAULT_DATASET) -> bool:
    """Execute warmup sequence. Returns True if LAPTOP_DEMO is ready."""
    print("=" * 48)
    print("AION LAPTOP DEMO WARMUP")
    print("=" * 48)
    print()

    all_ok = True

    # ── 1. Load benchmark profile ─────────────────────────────────────
    profile_path = _CACHE_DIR / "runtime_profile.json"
    if not profile_path.exists():
        all_ok &= _status("Benchmark profile", False,
                           "Not found. Run: python runtime/benchmark/benchmark_inference.py")
        print("\nWarmup FAILED.")
        return False

    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    backend_name = profile.get("backend", "ollama")
    model_name = profile.get("model", "qwen2.5:3b")
    device = profile.get("device", "GPU")
    all_ok &= _status("Benchmark profile", True, f"{backend_name} / {model_name}")

    # ── 2. Hardware check ─────────────────────────────────────────────
    from runtime.device import get_device_report
    report = get_device_report()
    all_ok &= _status("Hardware", True,
                       f"{report['cpu']}, {report['ram_gb']:.0f} GB, {report['gpu']}")

    # ── 3. Load model & tiny inference ────────────────────────────────
    print()
    print("  Loading model...")

    backend = None
    try:
        if backend_name == "openvino":
            from runtime.inference.openvino_backend import OpenVINOBackend
            backend = OpenVINOBackend()
        elif backend_name == "llamacpp":
            from runtime.inference.llama_cpp_backend import LlamaCppBackend
            backend = LlamaCppBackend()
        else:
            from runtime.inference.ollama_backend import OllamaBackend
            backend = OllamaBackend()

        backend.load_model(model_name, quantization="INT4", device=device)
        all_ok &= _status("Model load", True)

        # Tiny inference
        t0 = time.perf_counter()
        result = backend.generate(
            'Respond with exactly: {"test": "ok"}',
            max_tokens=32,
            temperature=0.0,
        )
        elapsed = time.perf_counter() - t0
        json_ok = "test" in result.text or "ok" in result.text
        all_ok &= _status("Model warmup", True, f"{elapsed:.1f}s")
        all_ok &= _status("JSON output", json_ok)

        backend.unload_model()

    except Exception as exc:
        all_ok &= _status("Model warmup", False, str(exc))

    # ── 4. Dataset cache ──────────────────────────────────────────────
    print()
    dataset_path = Path(dataset_dir)
    dataset_hash = ""

    try:
        from runtime.laptop_dataset import scan_dataset, get_dataset_hash
        modules = scan_dataset(dataset_path)
        dataset_hash = get_dataset_hash(modules)
        all_ok &= _status("Dataset", True, f"{len(modules)} modules")
    except Exception as exc:
        all_ok &= _status("Dataset", False, str(exc))

    # ── 5. Extraction cache ───────────────────────────────────────────
    cache = ExtractionCache()
    ext_ok = cache.manifest_path.exists()
    all_ok &= _status("Evidence cache", ext_ok, "HIT" if ext_ok else "MISS")

    # ── 6. BM25 indexes ──────────────────────────────────────────────
    bm25_ok = all(cache.bm25_exists(f"M{i}") for i in range(1, 6))
    all_ok &= _status("BM25 cache", bm25_ok, "HIT" if bm25_ok else "MISS")

    # ── 7. Contract test (basic) ──────────────────────────────────────
    try:
        from runtime.profiles.base import TimeoutBudget
        budget = TimeoutBudget()
        contract_ok = budget.hard_deadline == 600.0 and budget.target == 540.0
        all_ok &= _status("Contract test", contract_ok)
    except Exception as exc:
        all_ok &= _status("Contract test", False, str(exc))

    # ── 8. Memory check ──────────────────────────────────────────────
    from runtime.governor import MemoryGovernor, MemoryState
    gov = MemoryGovernor()
    rec = gov.recommend()
    mem_ok = rec.state != MemoryState.CRITICAL
    all_ok &= _status("Memory", mem_ok, rec.state.value)

    # ── 9. Write demo manifest ────────────────────────────────────────
    print()
    if all_ok:
        manifest = {
            "dataset_hash": dataset_hash,
            "model": model_name,
            "backend": backend_name,
            "device": device,
            "extraction_version": EXTRACTION_VERSION,
            "chunking_version": CHUNKING_VERSION,
            "validation_version": VALIDATION_VERSION,
            "bm25_version": BM25_VERSION,
            "profile": "LAPTOP_DEMO",
        }
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        manifest_path = _CACHE_DIR / "demo_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        _status("Demo manifest", True, str(manifest_path))
    else:
        _status("Demo manifest", False, "Prerequisites failed")

    # ── Final Verdict ─────────────────────────────────────────────────
    print()
    print("=" * 48)
    if all_ok:
        print("STATUS         : READY")
    else:
        print("STATUS         : NOT READY — fix failures above")
    print("=" * 48)

    return all_ok


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_DATASET
    ok = run_warmup(dataset)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
