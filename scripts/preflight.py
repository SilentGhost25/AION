#!/usr/bin/env python3
"""
Preflight gate for AION v3 server environment.
Validates: env vars -> GPU VRAM -> ports -> vLLM model -> schema -> uploads -> Groq.
Exit 0 = safe to run. Exit 1 = red, do not proceed.
"""
from __future__ import annotations

import json
import os
import pathlib
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Callable, List

ROOT = pathlib.Path(__file__).resolve().parents[1]
FAILS: List[str] = []
WARNS: List[str] = []


def check(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        print(f"  [PASS] {name}")
    except Exception as e:
        FAILS.append(f"{name}: {e}")
        print(f"  [FAIL] {name}: {e}")


def warn(name: str, fn: Callable[[], None]) -> None:
    try:
        fn()
        print(f"  [PASS] {name}")
    except Exception as e:
        WARNS.append(f"{name}: {e}")
        print(f"  [WARN] {name}: {e}")


# ---- Phase 0: env & ports ----
def env_required() -> None:
    required = ["AION_BACKEND", "AION_LLM_HOST", "AION_MODEL"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"missing required env vars: {missing}")
    backend = os.environ["AION_BACKEND"].lower().strip()
    if backend != "vllm":
        raise RuntimeError(f"AION_BACKEND={backend} (expected 'vllm')")


def port_free(port: int, allow_listening: bool = False) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", port))
        if not allow_listening:
            raise RuntimeError(f"port {port} already in use by another process")
    except (ConnectionRefusedError, socket.timeout):
        if allow_listening:
            raise RuntimeError(f"nothing listening on expected port {port}")
    finally:
        s.close()


def gpu_available() -> None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        WARNS.append("nvidia-smi not found in PATH (running on CPU/laptop?)")
        return
    if out.returncode != 0:
        raise RuntimeError(f"nvidia-smi error: {out.stderr.strip() or 'failed'}")
    free_mb = int(out.stdout.strip().splitlines()[0])
    if free_mb < 16_000:
        raise RuntimeError(f"only {free_mb} MiB free VRAM (<16 GB available)")
    print(f"         GPU VRAM free: {free_mb} MiB")


# ---- Phase 1: vLLM alive, model exact, schema support ----
def vllm_alive() -> None:
    host = (os.environ.get("AION_LLM_HOST") or "http://127.0.0.1:8000").rstrip("/")
    url = f"{host}/v1/models"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=4) as r:
        data = json.load(r)
    ids = [m["id"] for m in data.get("data", [])]
    want = os.environ.get("AION_MODEL", "")
    if want not in ids:
        raise RuntimeError(f"AION_MODEL={want!r} not served. Available on engine: {ids}")
    print(f"         Engine model verified: {want}")


def vllm_schema_support() -> None:
    """Probe whether the engine accepts json_schema response_format."""
    host = (os.environ.get("AION_LLM_HOST") or "http://127.0.0.1:8000").rstrip("/")
    body = json.dumps({
        "model": os.environ.get("AION_MODEL", ""),
        "messages": [{"role": "user", "content": "reply with ok"}],
        "max_tokens": 8,
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "probe",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
            },
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as res:
            if res.status != 200:
                WARNS.append(f"vLLM probe returned HTTP {res.status}")
    except urllib.error.HTTPError as e:
        if e.code == 400:
            WARNS.append("vLLM engine rejects json_schema guided decoding; fallback parser will be used")
        else:
            raise


# ---- Phase 2: uploads + extraction ----
def uploads_present() -> None:
    d = ROOT / "workspace" / "uploads"
    if not d.exists():
        raise RuntimeError(f"Workspace uploads dir {d} does not exist")
    pdfs = list(d.glob("*/original.pdf"))
    if not pdfs:
        raise RuntimeError(f"No original.pdf files found under {d}")
    print(f"         Found {len(pdfs)} uploaded PDF(s). Sample ID: {pdfs[0].parent.name}")


def sample_pdf_extractable() -> None:
    d = ROOT / "workspace" / "uploads"
    pdfs = list(d.glob("*/original.pdf"))
    if not pdfs:
        return
    pdf = pdfs[0]
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf)
        chars = sum(len(p.get_text()) for p in doc)
        doc.close()
        if chars < 300:
            WARNS.append(f"{pdf.name} has only {chars} extractable chars (scanned PDF / OCR required)")
        else:
            print(f"         Sample PDF extractable text: {chars} characters")
    except ImportError:
        WARNS.append("PyMuPDF (fitz) not importable in current python environment")


# ---- Phase 3: Groq ----
def groq_reachable() -> None:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY environment variable is unset")
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            if r.status != 200:
                WARNS.append(f"Groq API returned HTTP {r.status}")
    except urllib.error.HTTPError as e:
        if e.code == 429:
            WARNS.append("Groq rate-limited (429) right now; evaluation will degrade to local mode")
        elif e.code == 401:
            raise RuntimeError("GROQ_API_KEY is invalid / unauthorized (401)")
        else:
            raise


def main() -> int:
    print("==================================================")
    print(" AION v3 Preflight Verification Gate")
    print("==================================================")
    print("\n[Phase 0: Environment & Ports]")
    check("env vars", env_required)
    check("GPU VRAM", gpu_available)
    check("port 8000 has vLLM engine", lambda: port_free(8000, allow_listening=True))
    check("port 8100 free for backend", lambda: port_free(8100, allow_listening=False))
    check("port 5174 free for frontend", lambda: port_free(5174, allow_listening=False))

    print("\n[Phase 1: vLLM & Model Resolution]")
    check("vLLM alive + model exact", vllm_alive)
    warn("vLLM json_schema guided decoding", vllm_schema_support)

    print("\n[Phase 2: Uploads & Document Extraction]")
    check("uploads present", uploads_present)
    warn("sample PDF extractable", sample_pdf_extractable)

    print("\n[Phase 3: External Auditor Reachability]")
    warn("Groq API reachable", groq_reachable)

    print("\n--------------------------------------------------")
    if FAILS:
        print(f"[ERROR] {len(FAILS)} hard failure(s). DO NOT proceed with pipeline run:")
        for f in FAILS:
            print(f"   * {f}")
        return 1

    print(f"[SUCCESS] Preflight passed ({len(WARNS)} warning(s)). Safe to start server & pipeline.")
    for w in WARNS:
        print(f"   * [WARN] {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
