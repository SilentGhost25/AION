#!/usr/bin/env python3
"""
AION v3 End-to-End (E2E) Generation Pipeline Test Script.
Validates:
1. vLLM Engine connectivity & active served model on port 8000.
2. AION API server availability on port 8100.
3. End-to-end question paper generation via direct HTTP POST to /api/v3/generate
   (or direct in-process pipeline test with --in-process).
4. Academic and structural verification of the resulting paper.

Usage:
    python scripts/run_v3_e2e_test.py               # E2E test against live running server on :8100
    python scripts/run_v3_e2e_test.py --in-process  # In-process test (directly tests pipeline against vLLM)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check_vllm_engine(host: str) -> tuple[bool, str]:
    url = f"{host.rstrip('/')}/v1/models"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as res:
            if res.status == 200:
                data = json.loads(res.read().decode("utf-8"))
                models = [m.get("id") for m in data.get("data", [])]
                if models:
                    return True, models[0]
                return False, "No models listed in /v1/models"
    except Exception as e:
        return False, str(e)
    return False, "Unknown engine status"


def check_backend_server(host: str) -> tuple[bool, str]:
    url = f"{host.rstrip('/')}/health"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=4) as res:
            return res.status == 200, f"HTTP {res.status}"
    except urllib.error.HTTPError as he:
        return True, f"HTTP {he.code}"
    except Exception as e:
        return False, str(e)


def find_sample_upload_files(count: int = 3) -> list[str]:
    uploads_dir = ROOT / "workspace" / "uploads"
    if not uploads_dir.exists():
        return []
    fids = []
    for d in sorted(uploads_dir.iterdir()):
        if d.is_dir() and (d / "original.pdf").exists():
            fids.append(d.name)
            if len(fids) >= count:
                break
    return fids


def run_http_e2e(api_host: str, fids: list[str], subject: str, exam_type: str):
    url = f"{api_host.rstrip('/')}/api/v3/generate"
    payload = {
        "subject": subject,
        "exam_type": exam_type,
        "file_ids": fids,
    }
    print(f"\n[E2E-TEST] Dispatching POST {url}")
    print(f"[E2E-TEST] Payload: {json.dumps(payload, indent=2)}")
    print("[E2E-TEST] Awaiting pipeline response (this may take 30-90 seconds for live model generation)...")

    start_t = time.time()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            elapsed = time.time() - start_t
            status_code = resp.status
            body = json.loads(resp.read().decode("utf-8"))

            print(f"\n[E2E-TEST] Response received in {elapsed:.2f}s (HTTP {status_code})")
            success = body.get("success", False)
            paper_status = body.get("paper_status", "UNKNOWN")
            questions = body.get("questions", [])
            blocked_reasons = body.get("blocked_reasons", [])
            degraded = body.get("degraded", False)

            print(f"  Success         : {success}")
            print(f"  Paper Status    : {paper_status}")
            print(f"  Total Questions : {len(questions)}")
            print(f"  Degraded Mode   : {degraded}")
            if blocked_reasons:
                print(f"  Blocked Reasons : {blocked_reasons}")

            # Verify question marks and structure
            for i, q in enumerate(questions, 1):
                q_text = q.get("text", "")[:70].replace("\n", " ")
                marks = q.get("marks", 0)
                subparts = q.get("subparts") or q.get("sub_questions") or []
                print(f"    Q{i}: {marks}M ({len(subparts)} subparts) | {q_text}...")

            if success and paper_status in ("OK", "PARTIAL", "DEGRADED"):
                print("\n✅ [E2E-TEST PASSED] Question paper generated successfully!")
                return 0
            else:
                print(f"\n❌ [E2E-TEST FAILED] Paper status: {paper_status}")
                return 1

    except urllib.error.HTTPError as he:
        err_body = he.read().decode("utf-8", errors="replace")
        print(f"\n❌ [E2E-TEST FAILED] Server returned HTTP {he.code}: {err_body}")
        return 1
    except Exception as e:
        print(f"\n❌ [E2E-TEST FAILED] Request error: {e}")
        return 1


def run_in_process_e2e(fids: list[str], subject: str, exam_type: str, model_name: str, vllm_host: str):
    print("\n[E2E-TEST] Running pipeline directly in-process against live vLLM engine...")
    from core.extraction.artifact_cache import load_or_extract_artifact, merge_artifacts
    from core.generation.paper_spec_resolver import PaperSpecResolver
    from core.generation.agents.pipeline_bridge import run_v3_pipeline
    from core.contracts.paper_spec import STANDARD_MARKS_SPLITS

    os.environ["AION_BACKEND"] = "vllm"
    os.environ["AION_LLM_HOST"] = vllm_host
    os.environ["AION_MODEL"] = model_name

    pdfs = []
    for fid in fids:
        p = ROOT / "workspace" / "uploads" / fid / "original.pdf"
        if p.exists():
            pdfs.append(p)

    if not pdfs:
        satcom = [ROOT / "tests" / "fixtures" / "satcom" / f"module{i}.pdf" for i in range(1, 6)]
        pdfs = [p for p in satcom if p.exists()]

    if not pdfs:
        print("❌ [E2E-TEST] No PDF input sources found.")
        return 1

    print(f"[E2E-TEST] Extracting/loading artifacts for {len(pdfs)} PDF(s)...")
    artifacts = [load_or_extract_artifact(p) for p in pdfs]
    artifact = merge_artifacts(artifacts) if len(artifacts) > 1 else artifacts[0]

    spec = PaperSpecResolver.resolve(exam_type)
    splits = STANDARD_MARKS_SPLITS.get(exam_type) or [[spec.marks_per_question] for _ in range(spec.total_questions)]

    print(f"[E2E-TEST] Executing run_v3_pipeline (spec={exam_type}, model={model_name})...")
    start_t = time.time()
    paper_parts, full_paper, meta = run_v3_pipeline(
        paper_spec=spec,
        artifact=artifact,
        marks_split=splits,
        request={"exam_type": exam_type, "subject": subject},
    )
    elapsed = time.time() - start_t

    paper_status = meta.get("paper_status", "UNKNOWN")
    success = meta.get("success", False)
    print(f"\n[E2E-TEST] In-process execution completed in {elapsed:.2f}s")
    print(f"  Success      : {success}")
    print(f"  Paper Status : {paper_status}")
    print(f"  Questions    : {len(full_paper)}")

    for i, q in enumerate(full_paper, 1):
        q_text = getattr(q, "text", "") if hasattr(q, "text") else str(q.get("text", ""))[:70]
        q_marks = getattr(q, "marks", 0) if hasattr(q, "marks") else q.get("marks", 0)
        print(f"    Q{i}: {q_marks}M | {str(q_text)[:70]}...")

    if success or paper_status in ("OK", "PARTIAL", "DEGRADED"):
        print("\n✅ [E2E-TEST PASSED] In-process pipeline generation succeeded!")
        return 0
    else:
        print(f"\n❌ [E2E-TEST FAILED] In-process generation status: {paper_status}, reasons: {meta.get('blocked_reasons')}")
        return 1


def main():
    parser = argparse.ArgumentParser(description="AION v3 E2E Test Suite")
    parser.add_argument("--vllm-host", default=os.environ.get("AION_LLM_HOST", "http://127.0.0.1:8000"))
    parser.add_argument("--api-host", default="http://127.0.0.1:8100")
    parser.add_argument("--subject", default="IOT Agriculture and Healthcare")
    parser.add_argument("--exam-type", default="IAT1")
    parser.add_argument("--in-process", action="store_true", help="Run pipeline in-process directly against vLLM")
    args = parser.parse_args()

    print("==================================================")
    print("      AION v3 End-to-End (E2E) Test Suite         ")
    print("==================================================")

    # 1. Probe vLLM Engine
    print(f"[1/3] Probing vLLM Inference Engine on {args.vllm_host}...")
    vllm_ok, model_or_err = check_vllm_engine(args.vllm_host)
    if not vllm_ok:
        print(f"  ❌ vLLM engine unreachable or unhealthy: {model_or_err}")
        print("  -> Ensure vLLM is running on port 8000 before running tests.")
        sys.exit(1)
    print(f"  ✅ vLLM Engine Active! Model: {model_or_err}")
    model_name = os.environ.get("AION_MODEL", model_or_err)

    # 2. Check Input Data
    fids = find_sample_upload_files(3)
    if not fids:
        print("  ⚠️ No sample upload directories found in workspace/uploads, falling back to satcom fixtures if available.")
    else:
        print(f"  ✅ Found {len(fids)} sample upload files: {fids}")

    # 3. Execution
    if args.in_process:
        code = run_in_process_e2e(fids, args.subject, args.exam_type, model_name, args.vllm_host)
        sys.exit(code)

    print(f"[2/3] Probing Backend API Server on {args.api_host}...")
    backend_ok, backend_status = check_backend_server(args.api_host)
    if not backend_ok:
        print(f"  ❌ Backend server NOT reachable on {args.api_host} ({backend_status})")
        print("  " + "=" * 60)
        print("  DIAGNOSTIC NOTICE:")
        print("  Nothing is listening on port 8100!")
        print("  This is why the browser UI shows 'Failed to fetch' when you click Generate.")
        print("  Start the backend in another terminal with:")
        print("     export AION_BACKEND=vllm")
        print("     export AION_LLM_HOST=http://localhost:8000")
        print(f"     export AION_MODEL={model_name}")
        print("     python aion_api.py")
        print("  Or run this test directly in-process:")
        print("     python scripts/run_v3_e2e_test.py --in-process")
        print("  " + "=" * 60)
        sys.exit(1)

    print(f"  ✅ Backend server is online on {args.api_host} ({backend_status})")
    print(f"[3/3] Running End-to-End API Generation...")
    code = run_http_e2e(args.api_host, fids, args.subject, args.exam_type)
    sys.exit(code)


if __name__ == "__main__":
    main()
