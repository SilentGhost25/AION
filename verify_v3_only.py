"""
verify_v3_only.py — Local verification that v3 is the only active path.

No GPU, no LLM, no API keys required. Uses static analysis, guard
invocation, and stub-driven pipeline execution.

Exit code 0 = all checks pass, v3 is unconditionally active.
Exit code 1 = at least one check failed, legacy path may still be reachable.
"""

from __future__ import annotations
import contextlib
import importlib
import io
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name: str, ok: bool, detail: str = ""):
    status = PASS if ok else FAIL
    line = f"[{status}] {name}"
    if detail and not ok:
        line += f" — {detail}"
    results.append((ok, name))
    print(line, flush=True)


def section(title: str):
    print(f"\n{'=' * 72}\n  {title}\n{'=' * 72}", flush=True)


# =============================================================================
# CHECK 1 — run_pipeline unconditionally delegates to v3
# =============================================================================

section("CHECK 1 — run_pipeline delegates to v3 (no flag check)")

try:
    import ast
    main_src = (REPO / "v0_1" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(main_src)
    run_pipe_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_pipeline":
            run_pipe_node = node
            break
    if not run_pipe_node:
        check("run_pipeline body extracted", False, "function not found")
    else:
        body_src = ast.get_source_segment(main_src, run_pipe_node) or ""
        has_v3 = "run_v3_pipeline" in body_src
        has_flag_check = "v3_enabled()" in body_src or "AION_ENABLE_V3_AGENTS" in body_src
        check("run_pipeline calls run_v3_pipeline", has_v3,
              "run_v3_pipeline not referenced")
        check("run_pipeline has no flag check", not has_flag_check,
              "flag check found in run_pipeline body")
except Exception as e:
    check("run_pipeline static analysis", False, str(e))


# =============================================================================
# CHECK 2 — SlotOrchestrator guard is active
# =============================================================================

section("CHECK 2 — SlotOrchestrator guard raises RuntimeError")

try:
    orch_src = (REPO / "core" / "generation" / "orchestrator.py").read_text(encoding="utf-8")
    has_guard = "Legacy SlotOrchestrator is disabled" in orch_src
    check("guard text present in orchestrator.py", has_guard)

    # Try to invoke the guard
    mod = importlib.import_module("core.generation.orchestrator")
    SlotOrch = getattr(mod, "SlotOrchestrator", None)
    if SlotOrch is None:
        check("SlotOrchestrator class importable", False, "class not found")
    else:
        # Try generate() with minimal args
        try:
            import inspect
            sig = inspect.signature(SlotOrch.__init__)
            try:
                instance = SlotOrch()
            except TypeError:
                req = [p for p in sig.parameters.values()
                       if p.name != "self" and p.default is p.empty]
                instance = SlotOrch(*[None] * len(req))

            try:
                instance.generate(None, None)
                check("SlotOrchestrator.generate() raises", False,
                      "no exception raised")
            except RuntimeError as e:
                raised_guard = "Legacy SlotOrchestrator is disabled" in str(e)
                check("SlotOrchestrator.generate() raises guard", raised_guard,
                      f"wrong error: {e}")
            except Exception as e:
                check("SlotOrchestrator.generate() raises RuntimeError",
                      False, f"raised {type(e).__name__}: {e}")
        except Exception as e:
            check("SlotOrchestrator instantiation", False, str(e))
except Exception as e:
    check("orchestrator guard check", False, str(e))


# =============================================================================
# CHECK 3 — v3 agents do not import legacy modules
# =============================================================================

section("CHECK 3 — v3 agents do not import legacy modules")

agents_dir = REPO / "core" / "generation" / "agents"
legacy_targets = [
    "demand_validator",
    "core.validation.linter",
    "from v0_1.generator",
    "from v0_1.orchestrator",
    "core.generation.orchestrator",
]

violations = []
if not agents_dir.exists():
    check("agents directory exists", False, str(agents_dir))
else:
    for f in agents_dir.rglob("*.py"):
        src = f.read_text(encoding="utf-8")
        for target in legacy_targets:
            if target in src:
                if "Legacy SlotOrchestrator is disabled" in src:
                    continue
                violations.append(f"{f.relative_to(REPO)} -> {target}")

check("no legacy imports in v3 agents", len(violations) == 0,
      f"violations: {violations}")


# =============================================================================
# CHECK 4 — Environment has no legacy override set
# =============================================================================

section("CHECK 4 — AION_ALLOW_LEGACY_ORCHESTRATOR is not set")

legacy_env = os.environ.get("AION_ALLOW_LEGACY_ORCHESTRATOR", "false").lower()
check("AION_ALLOW_LEGACY_ORCHESTRATOR disabled",
      legacy_env not in ("true", "1", "yes"),
      f"value = {legacy_env}")

v3_env = os.environ.get("AION_ENABLE_V3_AGENTS", "false").lower()
print(f"  (info) AION_ENABLE_V3_AGENTS = {v3_env} — irrelevant now, "
      f"v3 is unconditional", flush=True)


# =============================================================================
# CHECK 5 — v3 pipeline runs end-to-end with mocks, emits agent lines,
#           and emits NO legacy markers
# =============================================================================

section("CHECK 5 — v3 pipeline runs with stubs (no legacy fallback)")

try:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "tests"))

    from core.generation.agents.dependency_factory import DependencyFactory
    from core.generation.agents.pipeline_bridge import run_v3_pipeline

    try:
        from tests.fixtures.v3_mock_callers import (
            MockLLMCaller, MockVLMCaller, MockAPICaller, MockSympyVerifier,
        )
        from tests.fixtures.v3_paper_specs import (
            IAT1_SPEC, IAT1_MARKS_SPLIT, make_fake_artifact,
        )
        fixtures_available = True
    except ImportError as e:
        check("test fixtures importable", False, str(e))
        fixtures_available = False

    if fixtures_available:
        factory = DependencyFactory(
            llm_caller_factory=lambda: MockLLMCaller(),
            vlm_caller_factory=lambda: MockVLMCaller(),
            api_caller_factory=lambda: MockAPICaller(mode="all_pass"),
            sympy_verifier_factory=lambda: MockSympyVerifier(result=True),
        )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            parts, full, meta = run_v3_pipeline(
                paper_spec=IAT1_SPEC,
                artifact=make_fake_artifact(5, subject_prefix="Verify"),
                marks_split=IAT1_MARKS_SPLIT,
                dependency_factory=factory,
            )
        output = buf.getvalue()

        agent_markers = ["[AGENT:planning]", "[AGENT:writing]",
                         "[AGENT:evaluation]", "[AGENT:refinement]",
                         "[AGENT:checking]"]
        for marker in agent_markers:
            present = marker in output
            check(f"{marker} emitted", present,
                  "agent line missing from output" if not present else "")

        legacy_markers = [
            "[ORCHESTRATOR] Slot",
            "[ORCHESTRATOR] Oscillation detected",
            "[EXPORT GATE FAIL-CLOSED] Slot",
            "[DEMAND]",
            "[TOPIC] Stripped header",
        ]
        for marker in legacy_markers:
            present = marker in output
            check(f"no legacy marker: {marker!r}", not present,
                  "legacy code path executed" if present else "")

        check("pipeline_success is True", meta.get("success") is True,
              f"meta = {meta.get('failure_code')}")
        check("paper_parts emitted", len(parts) == IAT1_SPEC.module_count,
              f"got {len(parts)} modules, expected {IAT1_SPEC.module_count}")
        check("full_paper emitted", len(full) == IAT1_SPEC.total_questions,
              f"got {len(full)} questions, expected {IAT1_SPEC.total_questions}")

except Exception as e:
    import traceback
    check("v3 pipeline execution", False, str(e))
    traceback.print_exc()


# =============================================================================
# SUMMARY
# =============================================================================

section("SUMMARY")
total = len(results)
passed = sum(1 for ok, _ in results if ok)
failed = total - passed
print(f"  Passed: {passed}/{total}")
if failed:
    print(f"  Failed: {failed}")
    print("\n  Failing checks:")
    for ok, name in results:
        if not ok:
            print(f"    - {name}")
    sys.exit(1)
else:
    print("  All checks passed.")
    print("\n  v3 is the only active pipeline. No legacy fallback detected.")
    sys.exit(0)
