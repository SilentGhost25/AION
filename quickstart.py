#!/usr/bin/env python3
"""
Diagnostic & Quickstart runner for AION v3 pipeline.
Bypasses Flask/SSE and runs the pipeline directly against real or fixture PDFs.
"""

import os
import json
import time
from pathlib import Path

# Align environment to active vLLM production engine
os.environ.setdefault("AION_BACKEND", "vllm")
os.environ.setdefault("AION_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
os.environ.setdefault("AION_LLM_HOST", "http://localhost:8000")
os.environ.setdefault("AION_ENABLE_V3_AGENTS", "true")

from core.extraction.artifact_cache import load_or_extract_artifact, merge_artifacts
from core.generation.agents.pipeline_bridge import run_v3_pipeline
from core.generation.paper_spec_resolver import PaperSpecResolver
from core.contracts.paper_spec import STANDARD_MARKS_SPLITS

root_dir = Path(__file__).resolve().parent

# Check for uploaded IoT PDFs, otherwise fallback to SatCom fixtures
iot_pdfs = [
    root_dir / "workspace" / "uploads" / "b56ada8d-c9a" / "original.pdf",
    root_dir / "workspace" / "uploads" / "e59d96e4-d0b" / "original.pdf",
    root_dir / "workspace" / "uploads" / "081fcfb8-bf2" / "original.pdf",
]
satcom_pdfs = [
    root_dir / "tests" / "fixtures" / "satcom" / f"module{i}.pdf"
    for i in range(1, 6)
]

if all(p.exists() for p in iot_pdfs):
    pdfs = iot_pdfs
    subject = "IOT Agriculture and Health care"
else:
    pdfs = [p for p in satcom_pdfs if p.exists()]
    subject = "Satellite Communication"

print(f"\n{'='*70}")
print(f"AION QUICKSTART DIAGNOSTIC — {subject}")
print(f"{'='*70}")
print(f"Loaded {len(pdfs)} input documents:")
for p in pdfs:
    print(f"  - {p}")

artifacts = [load_or_extract_artifact(p) for p in pdfs]
artifact = merge_artifacts(artifacts) if len(artifacts) > 1 else artifacts[0]
spec = PaperSpecResolver.resolve("IAT1")
split = STANDARD_MARKS_SPLITS.get("IAT1", [
    [10], [10], [6, 4], [6, 4], [6, 4], [6, 4], [10], [10], [10], [10]
])

print("\n=== STEP 1: run_v3_pipeline ===")
t0 = time.time()
paper_parts, full_paper, meta = run_v3_pipeline(
    paper_spec=spec,
    artifact=artifact,
    marks_split=split,
    request={"exam_type": "IAT1", "subject": subject},
)
elapsed = time.time() - t0

print(f"  elapsed         : {elapsed:.2f}s")
print(f"  pipeline_success: {meta.get('success')}")
print(f"  paper_status    : {meta.get('paper_status')}")
print(f"  paper_parts     : {len(paper_parts)} modules")
print(f"  full_paper      : {len(full_paper)} questions")

if meta.get("failure_code"):
    print(f"  failure_code    : {meta.get('failure_code')}")
    print(f"  failure_detail  : {meta.get('failure_detail')}")

# Write raw output for inspection
out_path = root_dir / "quickstart_output.json"
out_path.write_text(json.dumps({
    "paper_parts": paper_parts,
    "full_paper": full_paper,
    "meta": meta,
}, indent=2, default=str), encoding="utf-8")
print(f"  written to      : {out_path}")

print("\n=== STEP 2: legacy_adapter output shape ===")
if full_paper:
    q = full_paper[0]
    print(f"  Slot ID         : {q.get('slot_id')}")
    print(f"  Top-level keys  : {sorted(list(q.keys()))}")
    print(f"  sub_questions   : {len(q.get('sub_questions', []))} items")
    print(f"  subQuestions    : {len(q.get('subQuestions', []))} items")
    print(f"  partition       : {q.get('partition')}")
    print(f"  marks           : {q.get('marks')}")
else:
    print("  [ERROR] full_paper is empty!")

print("\n=== STEP 3: Formatter Contract Check ===")
for q in full_paper[:4]:
    subs = q.get("subQuestions") or q.get("sub_questions") or []
    part_marks = [int(s.get("marks", 0)) for s in subs]
    status = "OK" if sum(part_marks) == q.get("marks") else "MISMATCH"
    print(f"  [{q.get('slot_id')}]: {part_marks} -> sum={sum(part_marks)} / expected={q.get('marks')} [{status}]")

print("\n=== STEP 4: Direct _format_paper simulation ===")
try:
    from aion_api import _format_paper, validate_final_paper_contract
    formatted = _format_paper(
        paper=paper_parts,
        subject=subject,
        exam_type="IAT1",
        mode="turbo",
        qa_report=meta.get("qa_report"),
    )
    mod_count = len(formatted.get("modules", []))
    print(f"  _format_paper   : SUCCESS ({mod_count} modules formatted)")
    try:
        validate_final_paper_contract(formatted, "IAT1")
        print(f"  Contract Gate   : PASS")
    except Exception as cg_err:
        print(f"  Contract Gate   : NOTICE — {cg_err}")
except Exception as fmt_err:
    print(f"  _format_paper   : FAILED with {fmt_err}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*70}\n")
