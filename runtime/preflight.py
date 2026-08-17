# runtime/preflight.py
"""AION Laptop Preflight — validates all prerequisites before generation.

Run as:
    python -m runtime.preflight

If this prints LAPTOP_FAST READY, generation may start.
Otherwise generation MUST NOT start.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.device import get_device_report
from runtime.governor import MemoryGovernor, MemoryState
from runtime.retrieval.cache import ExtractionCache, EXTRACTION_VERSION, CHUNKING_VERSION, VALIDATION_VERSION, BM25_VERSION

# Default dataset path
_DEFAULT_DATASET = r"C:\Users\Tarun J\Downloads\New Dataset\IAI"
_CACHE_DIR = Path(".aion_cache")


def _check(label: str, passed: bool, detail: str = "") -> bool:
    """Print a single preflight check result."""
    status = "PASS" if passed else "FAIL"
    msg = f"  {label:<24} {status}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return passed


def run_preflight(dataset_dir: str = _DEFAULT_DATASET) -> bool:
    """Execute all preflight checks. Returns True if all pass."""
    print("=" * 60)
    print("AION LAPTOP PREFLIGHT")
    print("=" * 60)
    print()

    all_pass = True

    # ── Hardware ──────────────────────────────────────────────────────
    report = get_device_report()

    all_pass &= _check("CPU", True, report["cpu"])
    all_pass &= _check("RAM", report["ram_gb"] >= 8.0, f"{report['ram_gb']:.1f} GB")

    gpu = report["gpu"]
    has_arc = "arc" in gpu.lower() or "intel" in gpu.lower()
    all_pass &= _check("Intel Arc", has_arc, gpu)

    print()

    # ── Backends ──────────────────────────────────────────────────────
    all_pass &= _check("OpenVINO", report["openvino_available"])

    ov_gpu = False
    if report["openvino_available"]:
        try:
            import openvino as ov
            core = ov.Core()
            devices = core.available_devices
            ov_gpu = "GPU" in devices
        except Exception:
            pass
    all_pass &= _check("OpenVINO GPU", ov_gpu)

    all_pass &= _check("llama.cpp", report["llamacpp_available"])

    # Vulkan check (best-effort)
    vulkan = False
    try:
        import subprocess
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True, text=True, timeout=5
        )
        vulkan = result.returncode == 0
    except Exception:
        pass
    _check("Vulkan", vulkan)  # Informational, not blocking

    all_pass &= _check("Ollama", report["ollama_available"])

    print()

    # ── Dataset ───────────────────────────────────────────────────────
    dataset_path = Path(dataset_dir)
    dataset_exists = dataset_path.is_dir()
    all_pass &= _check("Dataset", dataset_exists, str(dataset_path))

    if dataset_exists:
        try:
            from runtime.laptop_dataset import scan_dataset
            modules = scan_dataset(dataset_path)
            for mid in sorted(modules.keys()):
                m = modules[mid]
                all_pass &= _check(mid, True, m.path.name)
        except Exception as exc:
            all_pass &= _check("Module scan", False, str(exc))
    else:
        for i in range(1, 6):
            all_pass &= _check(f"M{i}", False, "Dataset not found")

    print()

    # ── Cache ─────────────────────────────────────────────────────────
    cache = ExtractionCache()
    extraction_cached = cache.manifest_path.exists()
    _check("Extraction cache", extraction_cached)

    bm25_cached = all(
        cache.bm25_exists(f"M{i}") for i in range(1, 6)
    ) if extraction_cached else False
    _check("BM25 indexes", bm25_cached)

    print()

    # ── Benchmark Profile ─────────────────────────────────────────────
    profile_path = _CACHE_DIR / "runtime_profile.json"
    profile_exists = profile_path.exists()
    model_name = "Not benchmarked"
    backend_name = "Not benchmarked"

    if profile_exists:
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)
            model_name = profile.get("model", "Unknown")
            backend_name = profile.get("backend", "Unknown")
        except Exception:
            profile_exists = False

    _check("Benchmark profile", profile_exists)
    print(f"  {'Model':<24} {model_name}")
    print(f"  {'Backend':<24} {backend_name}")

    print()

    # ── Memory Governor ───────────────────────────────────────────────
    gov = MemoryGovernor()
    rec = gov.recommend()
    mem_ok = rec.state != MemoryState.CRITICAL
    all_pass &= _check("Memory budget", mem_ok, rec.state.value)
    print(f"  {'Context':<24} {rec.context_length}")
    print(f"  {'Top-K':<24} {rec.retrieval_top_k}")

    # ── Final Verdict ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    if all_pass:
        print("LAPTOP_FAST READY")
    else:
        print("LAPTOP_FAST NOT READY — fix failures above")
    print("=" * 60)

    return all_pass


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_DATASET
    ok = run_preflight(dataset)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
