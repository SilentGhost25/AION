# runtime/__main__.py
"""Allow `python -m runtime` to show available commands."""

print("""
AION Runtime Commands:
    python -m runtime.preflight         — Validate laptop prerequisites
    python -m runtime.warmup            — Warm up caches for LAPTOP_DEMO
    python runtime/benchmark/benchmark_inference.py   — Benchmark backends
    python runtime/benchmark/benchmark_pipeline.py    — Full pipeline benchmark
""")
