import sys
import traceback

print("=" * 60)
print("AION Pre-flight Diagnostic")
print("=" * 60)

issues = []
warnings = []

# 1. Core imports
print("\n[1/8] Core imports...")
modules = [
    ("flask",                        "Flask"),
    ("requests",                     "requests"),
    ("v0_1.llm",                     "RobustLLMCaller"),
    ("v0_1.generator",               "get_vtu_vibe_question"),
    ("v0_1.segmenter",               "segment_document"),
    ("v0_1.extractor",               "extract"),
    ("v0_1.main",                    "run_pipeline"),
    ("v0_1.paper_validator",         "PaperValidator"),
    ("v0_1.paper_formatter",         "get_preview_html"),
    ("core.config.production_model", "get_production_model"),
]
for mod, attr in modules:
    try:
        m = __import__(mod, fromlist=[attr])
        getattr(m, attr)
        print(f"  OK  {mod}.{attr}")
    except Exception as e:
        print(f"  ERR {mod}.{attr}: {e}")
        issues.append(f"{mod}: {e}")

# 2. Model authority
print("\n[2/8] Model authority...")
try:
    from core.config.production_model import get_production_model, get_resolution_info
    model = get_production_model()
    info  = get_resolution_info()
    src   = info.get("source", "?")
    dev   = info.get("device", "?")
    print(f"  Model  : {model}")
    print(f"  Source : {src}")
    print(f"  Device : {dev}")
    if model in ("qwen2.5:1.5b", "aion", ""):
        warnings.append(f"Model {model!r} may not be installed on server")
except Exception as e:
    issues.append(f"Model authority: {e}")
    print(f"  ERR: {e}")

# 3. Ollama connectivity
print("\n[3/8] Ollama connectivity...")
try:
    import requests as req
    r      = req.get("http://127.0.0.1:11434/api/tags", timeout=5)
    models = [m["name"] for m in r.json().get("models", [])]
    print(f"  Ollama running. Models: {models}")
    from core.config.production_model import get_production_model
    expected = get_production_model()
    found    = any(expected in m for m in models)
    if not found:
        issues.append(f"Expected model {expected!r} not in Ollama: {models}")
        print(f"  ERR: {expected} not found in Ollama")
    else:
        print(f"  OK  {expected} is available")
except Exception as e:
    issues.append(f"Ollama: {e}")
    print(f"  ERR: {e}")

# 4. Pipeline contracts
print("\n[4/8] Pipeline contracts...")
try:
    from v0_1.contracts import (
        RawFile, ExtractionResult, CleanedContent,
        ChunkedContent, PipelineHealth, ContractViolation
    )
    h = PipelineHealth()
    h.deduct(10, "test")
    assert h.score == 90
    print("  OK  contracts")
except Exception as e:
    warnings.append(f"Contracts: {e}")
    print(f"  WARN: {e}")

# 5. aion_api.py internal consistency
print("\n[5/8] aion_api.py consistency...")
try:
    import aion_api
    import inspect
    sig    = inspect.signature(aion_api._format_paper)
    params = list(sig.parameters.keys())
    print(f"  _format_paper params: {params}")
    if "subject" not in params:
        issues.append("_format_paper missing subject parameter")
    if hasattr(aion_api, "_enforce_marks"):
        print("  OK  _enforce_marks present")
    else:
        issues.append("_enforce_marks not defined in aion_api")
        print("  ERR: _enforce_marks missing")
except Exception as e:
    issues.append(f"aion_api: {e}")
    print(f"  ERR: {e}")

# 6. run_pipeline signature
print("\n[6/8] run_pipeline signature...")
try:
    from v0_1.main import run_pipeline
    import inspect
    sig    = inspect.signature(run_pipeline)
    params = list(sig.parameters.keys())
    print(f"  run_pipeline params: {params}")
    for p in ["file_path", "exam_type", "difficulty", "mode"]:
        if p not in params:
            issues.append(f"run_pipeline missing param: {p}")
            print(f"  ERR: missing param {p}")
        else:
            print(f"  OK  {p}")
except Exception as e:
    issues.append(f"run_pipeline: {e}")
    print(f"  ERR: {e}")

# 7. Backend health endpoint
print("\n[7/8] Backend health endpoint...")
try:
    import requests as req
    r = req.get("http://127.0.0.1:8100/api/v1/health", timeout=5)
    if r.status_code == 200:
        d     = r.json()
        model = d.get("resolved_model", d.get("active_model", "?"))
        print(f"  OK  status={d.get('status')} model={model}")
    else:
        warnings.append(f"Health endpoint returned {r.status_code}")
        print(f"  WARN: {r.status_code}")
except Exception as e:
    warnings.append(f"Health endpoint unreachable: {e}")
    print(f"  WARN: {e}")

# 8. Required directories
print("\n[8/8] Required directories...")
from pathlib import Path
for d in ["workspace/uploads", "workspace/cache", "logs", "generated_papers", "templates"]:
    p = Path(d)
    if p.exists():
        print(f"  OK  {d}")
    else:
        p.mkdir(parents=True, exist_ok=True)
        print(f"  CREATED {d}")

# Summary
print("\n" + "=" * 60)
print(f"ISSUES   : {len(issues)}")
print(f"WARNINGS : {len(warnings)}")
print("=" * 60)
for i in issues:
    print(f"  ERROR: {i}")
for w in warnings:
    print(f"  WARN:  {w}")
if not issues:
    print("\nAll checks passed. Safe to run.")
