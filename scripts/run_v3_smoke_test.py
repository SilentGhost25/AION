#!/usr/bin/env python3
"""
Phase 5 — Manual smoke test for v3 on real LLM/VLM/API callers.

Usage:
    python scripts/run_v3_smoke_test.py --subject satcom --pdf path/to/pdf.pdf
    python scripts/run_v3_smoke_test.py --subject satcom --mock    # mock callers

This is a diagnostic tool, not a CI test. It runs the real pipeline
and prints a report. Use it to verify the flag flips cleanly on
production input before enabling AION_ENABLE_V3_AGENTS globally.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args():
    p = argparse.ArgumentParser(description="Run v3 pipeline smoke test")
    p.add_argument("--subject", required=True, help="Subject name (for logging)")
    p.add_argument("--pdf", required=False, help="Path to PDF (single or comma-separated)")
    p.add_argument("--exam-type", default="IAT1", help="IAT1 | IAT2 | ELECTIVE_3MOD")
    p.add_argument("--split-mode", choices=["standard", "uniform"], default="standard",
                   help="Marks partition mode: standard (VTU 10, 6+4 mix) or uniform (pure 10s)")
    p.add_argument("--mock", action="store_true",
                   help="Use mock callers instead of real LLM/VLM")
    return p.parse_args()


def main():
    args = parse_args()

    if args.mock:
        from tests.fixtures.v3_mock_callers import (
            MockAPICaller, MockLLMCaller, MockSympyVerifier, MockVLMCaller,
        )
        from tests.fixtures.v3_paper_specs import (
            IAT1_SPEC, IAT1_MARKS_SPLIT, IAT2_MARKS_SPLIT, make_fake_artifact,
            ELECTIVE_3MOD_SPEC, ELECTIVE_3MOD_MARKS_SPLIT,
        )
        from core.generation.agents.dependency_factory import DependencyFactory
        from core.generation.agents.pipeline_bridge import run_v3_pipeline

        spec = IAT1_SPEC if args.exam_type == "IAT1" else ELECTIVE_3MOD_SPEC
        if args.split_mode == "standard":
            splits_by_exam = {
                "IAT1": IAT1_MARKS_SPLIT,
                "IAT2": IAT2_MARKS_SPLIT,
                "ELECTIVE_3MOD": ELECTIVE_3MOD_MARKS_SPLIT,
            }
            split = splits_by_exam.get(args.exam_type, IAT1_MARKS_SPLIT)
        else:
            split = [[spec.marks_per_question] for _ in range(spec.total_questions)]
        artifact = make_fake_artifact(spec.module_count, subject_prefix=args.subject)

        factory = DependencyFactory(
            llm_caller_factory=lambda: MockLLMCaller(),
            vlm_caller_factory=lambda: MockVLMCaller(),
            api_caller_factory=lambda: MockAPICaller(mode="all_pass"),
            sympy_verifier_factory=lambda: MockSympyVerifier(result=True),
        )
    else:
        if not args.pdf:
            print("ERROR: --pdf required unless --mock is passed")
            sys.exit(2)

        from core.extraction.artifact_cache import load_or_extract_artifact, merge_artifacts
        from core.generation.paper_spec_resolver import resolve_paper_spec
        from core.generation.agents.dependency_factory import (
            DependencyFactory,
            default_llm_caller_factory,
            default_vlm_caller_factory,
            default_api_caller_factory,
            default_sympy_verifier_factory,
        )
        from core.generation.agents.pipeline_bridge import run_v3_pipeline
        from tests.fixtures.v3_paper_specs import (
            IAT1_MARKS_SPLIT,
            IAT2_MARKS_SPLIT,
            ELECTIVE_3MOD_MARKS_SPLIT,
        )

        spec = resolve_paper_spec(args.exam_type)
        pdfs = [Path(p.strip()) for p in args.pdf.split(",")]
        artifacts = [load_or_extract_artifact(p) for p in pdfs]
        artifact = merge_artifacts(artifacts) if len(artifacts) > 1 else artifacts[0]

        if args.split_mode == "standard":
            splits_by_exam = {
                "IAT1": IAT1_MARKS_SPLIT,
                "IAT2": IAT2_MARKS_SPLIT,
                "ELECTIVE_3MOD": ELECTIVE_3MOD_MARKS_SPLIT,
            }
            split = splits_by_exam.get(
                args.exam_type,
                [[spec.marks_per_question] for _ in range(spec.total_questions)],
            )
        else:
            split = [[spec.marks_per_question] for _ in range(spec.total_questions)]

        factory = DependencyFactory(
            llm_caller_factory=default_llm_caller_factory,
            vlm_caller_factory=default_vlm_caller_factory,
            api_caller_factory=default_api_caller_factory,
            sympy_verifier_factory=default_sympy_verifier_factory,
        )

    print(f"\n{'='*70}")
    print(f"V3 SMOKE TEST — {args.subject} ({args.exam_type})")
    print(f"{'='*70}")

    t0 = time.time()
    parts, full, meta = run_v3_pipeline(
        paper_spec=spec,
        artifact=artifact,
        marks_split=split,
        dependency_factory=factory,
    )
    elapsed = time.time() - t0

    print(f"\n[RESULT]")
    print(f"  elapsed         : {elapsed:.2f}s")
    print(f"  pipeline_success: {meta['success']}")
    print(f"  paper_status    : {meta.get('paper_status')}")
    print(f"  modules         : {len(parts)}")
    print(f"  questions       : {len(full)}")
    if meta.get("failure_code"):
        print(f"  failure_code    : {meta['failure_code']}")
        print(f"  failure_detail  : {meta.get('failure_detail')}")
    if meta.get("factory_notes"):
        print(f"  factory_notes   : {meta['factory_notes']}")

    qa = meta.get("qa_report") or {}
    print(f"\n[QA REPORT]")
    print(f"  slot_count      : {qa.get('slot_count')}")
    print(f"  resolved_slots  : {qa.get('resolved_slots')}")
    print(f"  unresolved      : {len(qa.get('unresolved_slots', []))}")
    print(f"  blocked_reasons : {qa.get('blocked_reasons')}")
    print(f"  degraded_reasons: {qa.get('degraded_reasons')}")

    print(f"\n[TELEMETRY]")
    for key in sorted(meta.get("telemetry", {}).keys()):
        print(f"  {key}: {meta['telemetry'][key]}")

    print(f"\n[QUESTIONS]")
    for q in full[:5]:
        vis = "V" if q.get("image_path") else "-"
        print(f"  [{q['slot_id']}] {vis} {q['marks']}M {q['bloom']} {q['co']}: "
              f"{q['question_text'][:60]}...")
    if len(full) > 5:
        print(f"  ... ({len(full) - 5} more)")

    print()


if __name__ == "__main__":
    main()
