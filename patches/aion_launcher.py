import os, sys, json, functools, importlib
aion_root = "/home/AIML1/AIQ/AION"
sys.path.insert(0, "/home/AIML1/AIQ/AION/patches")
sys.path.insert(0, aion_root)
import aion_hotfixes

# Patch LLM
for m in ("v0_1.llm", "core.generation.robust_llm_caller"):
    try:
        mod = importlib.import_module(m)
        if hasattr(mod, "RobustLLMCaller"):
            mod.RobustLLMCaller.call = aion_hotfixes.wrap_llm_call(mod.RobustLLMCaller.call)
    except: pass

# Patch CleanedDocument
for m in ("core.preprocessing.document_cleaner", "v0_1.schemas"):
    try:
        mod = importlib.import_module(m)
        if hasattr(mod, "CleanedDocument"):
            aion_hotfixes.patch_cleaned_document_cls(mod.CleanedDocument)
    except: pass

import aion_api

# Hook enforce_marks
orig_enforce = getattr(aion_api, "_enforce_marks", None)
if orig_enforce:
    @functools.wraps(orig_enforce)
    def new_enforce(paper, *a, **kw):
        paper = aion_hotfixes.transform_paper_structure(paper)
        try: return orig_enforce(paper, *a, **kw)
        except: return paper
    aion_api._enforce_marks = new_enforce

# Hook format_paper
orig_format = getattr(aion_api, "_format_paper", None)
if orig_format:
    @functools.wraps(orig_format)
    def new_format(paper, *a, **kw):
        paper = aion_hotfixes.transform_paper_structure(paper)
        return orig_format(paper, *a, **kw)
    aion_api._format_paper = new_format

@aion_api.app.before_request
def capture():
    from flask import request
    try:
        # Only process generation endpoints
        if request.path not in ("/api/generate", "/api/generate/stream", "/api/generate/vllm"):
            return

        data = request.get_json(silent=True) or {}
        exam = data.get("exam_type") or data.get("exam") or "IAT1"
        raw = (
            data.get("marks_distribution") or data.get("marksDistribution") or
            data.get("marks_split") or data.get("marksSplit") or
            data.get("mark_splits") or data.get("markSplits") or
            data.get("sub_question_marks")
        )

        if raw:
            split = aion_hotfixes.parse_desired_split(raw, exam)
            if split:
                aion_hotfixes.set_active_desired_split(split, exam)
                aion_hotfixes.patch_partition_table()
                print(f"[USER-CHOICE] Applied marks split: {split} for {exam}", flush=True)
            # if split is None, do NOT set anything
        else:
            pass
    except Exception as e:
        print(f"[ERROR] capture hook error: {e}")

if __name__ == "__main__":
    print("="*70)
    print("  AION - Surgical Partition Patcher ACTIVE")
    print("="*70)
    aion_api.app.run(host="0.0.0.0", port=8100, debug=False)
