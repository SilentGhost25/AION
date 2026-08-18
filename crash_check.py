import re
import os
from pathlib import Path

print("=" * 55)
print("AION Crash Risk Assessment")
print("=" * 55)

critical = []
high     = []
medium   = []

# -- Check 1: Token truncation in generator --------------------
print("\n[1] Checking num_predict / max_tokens limits...")
gen_file = Path("v0_1/generator.py")
if gen_file.exists():
    content = gen_file.read_text(errors="ignore")
    matches = re.findall(r"num_predict|_tokens_for_marks", content)
    for m in matches:
        val = int(m)
        if val < 300:
            critical.append(f"generator.py: num_predict={val} too low — sentences will be cut")
            print(f"  CRITICAL: num_predict={val} will truncate questions")
        else:
            print(f"  OK: num_predict={val}")
    if not matches:
        print("  WARN: num_predict not found in generator.py")
        high.append("generator.py: no num_predict — using Ollama default (may truncate)")

# -- Check 2: Chunk truncation in retriever --------------------
print("\n[2] Checking chunk truncation...")
for fname in ["v0_1/retriever.py", "v0_1/grounding_gate.py", "v0_1/generator.py"]:
    p = Path(fname)
    if not p.exists():
        continue
    content = p.read_text(errors="ignore")
    for pattern, risk in [
        (r'chunk\[:(\d+)\]',     "chunk sliced at"),
        (r'text\[:(\d+)\]',      "text sliced at"),
        (r'content\[:(\d+)\]',   "content sliced at"),
        (r'context\[:(\d+)\]',   "context sliced at"),
    ]:
        for m in re.finditer(pattern, content):
            val = int(m.group(1))
            if val < 500:
                critical.append(f"{fname}: {risk} {val} — retrieval chunks truncated mid-sentence")
                print(f"  CRITICAL: {fname}: {risk} {val}")
            else:
                print(f"  OK: {fname}: {risk} {val}")

# -- Check 3: Sentence completion check -----------------------
print("\n[3] Checking for sentence clipping...")
for fname in ["v0_1/generator.py", "v0_1/cleaner.py", "v0_1/critic.py"]:
    p = Path(fname)
    if not p.exists():
        continue
    content = p.read_text(errors="ignore")
    for pattern, desc in [
        (r'split\(["\']\\.\)',                  "split on period"),
        (r'\\[:\\s*([1-9]\\d?)\\s*\\]',                "short slice"),
        (r'\.rstrip\(["\'][.,;]',               "rstrip punctuation"),
        (r'truncate|clip|limit.*question',       "truncation keyword"),
    ]:
        if re.search(pattern, content, re.IGNORECASE):
            high.append(f"{fname}: {desc} found — may clip sentences")
            print(f"  HIGH: {fname}: {desc}")

# -- Check 4: result_sent defined before finally ---------------
print("\n[4] Checking result_sent placement...")
api = Path("aion_api.py")
if api.exists():
    lines = api.read_text(errors="ignore").splitlines()
    stream_line  = next((i for i, l in enumerate(lines) if "def stream():" in l), None)
    finally_line = next((i for i, l in enumerate(lines) if "if not result_sent" in l), None)
    init_line    = next((i for i, l in enumerate(lines) if "result_sent = False" in l and
                         (stream_line or 0) < i < (finally_line or 9999)), None)

    if stream_line and finally_line and init_line:
        if init_line < finally_line:
            print(f"  OK: result_sent initialized at L{init_line+1}, finally at L{finally_line+1}")
        else:
            critical.append("result_sent initialized AFTER finally block — will crash")
            print(f"  CRITICAL: result_sent at L{init_line+1} is after finally at L{finally_line+1}")
    else:
        high.append("Could not verify result_sent placement")
        print(f"  WARN: stream={stream_line} finally={finally_line} init={init_line}")

# -- Check 5: _enforce_marks called ---------------------------
print("\n[5] Checking _enforce_marks is called...")
if api.exists():
    content = api.read_text(errors="ignore")
    if "_enforce_marks" in content:
        calls = [l.strip() for l in content.splitlines() if "_enforce_marks(" in l]
        defns = [l.strip() for l in content.splitlines() if "def _enforce_marks" in l]
        print(f"  Definition : {defns[0] if defns else 'MISSING'}")
        print(f"  Call sites : {len(calls)}")
        for c in calls:
            print(f"    {c[:80]}")
    else:
        critical.append("_enforce_marks not found in aion_api.py — marks will be wrong")
        print("  CRITICAL: _enforce_marks missing")

# -- Check 6: include_visual removed from run_pipeline call ---
print("\n[6] Checking run_pipeline call arguments...")
if api.exists():
    content = api.read_text(errors="ignore")
    lines   = content.splitlines()
    in_call = False
    for i, line in enumerate(lines):
        if "_run_pipe(" in line or "run_pipeline(" in line:
            in_call = True
        if in_call:
            if "include_visual" in line:
                val = line.strip()
                if "False" in val or "false" in val:
                    print(f"  OK: include_visual=False at L{i+1}")
                else:
                    critical.append(f"include_visual passed as non-False at L{i+1}: {val}")
                    print(f"  CRITICAL: {val}")
            if ")" in line and in_call and i > 420:
                in_call = False
                break

# -- Check 7: SSE parser has currentEvent ---------------------
print("\n[7] Checking SSE parser...")
step2 = Path("frontend/artifacts/qp-generator/src/components/wizard/Step2Rules.tsx")
if step2.exists():
    content = step2.read_text(errors="ignore")
    if "currentEvent" in content:
        print("  OK: currentEvent tracking present")
    else:
        critical.append("SSE parser missing currentEvent — paper will not display")
        print("  CRITICAL: currentEvent missing from SSE parser")
    if "__aionLastPaper" in content:
        print("  OK: __aionLastPaper storage present")
    else:
        high.append("__aionLastPaper missing — paper assembly will use fallback")
        print("  HIGH: __aionLastPaper missing")

# -- Summary ---------------------------------------------------
print("\n" + "=" * 55)
print(f"CRITICAL : {len(critical)}")
print(f"HIGH     : {len(high)}")
print(f"MEDIUM   : {len(medium)}")
print("=" * 55)

for x in critical:
    print(f"  [CRITICAL] {x}")
for x in high:
    print(f"  [HIGH]     {x}")
for x in medium:
    print(f"  [MEDIUM]   {x}")

if not critical and not high:
    print("\nNo critical issues found. Ready to deploy.")
elif not critical:
    print("\nNo critical blockers. High items are improvements.")
else:
    print("\nFix critical items before deploying.")



