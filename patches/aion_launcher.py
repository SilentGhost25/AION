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
        data = request.get_json(silent=True) or {}
        print(f"\n{'='*70}\n[REQUEST] {request.endpoint}\n{json.dumps(data, indent=2)}\n{'='*70}\n")
        
        exam = data.get("exam_type") or data.get("exam") or "IAT1"
        raw = data.get("marks_distribution") or data.get("marksDistribution") or data.get("marks_split")
        
        if raw:
            split = aion_hotfixes.parse_desired_split(raw, exam)
        else:
            split = [10, 10] if "SEE" in exam.upper() else [5, 5]
        
        aion_hotfixes.set_active_desired_split(split, exam)
        aion_hotfixes.patch_partition_table()
        
        print(f"\n{'='*70}\n[✓ USER CHOICE] Exam: {exam} | Marks: {split}\n{'='*70}\n")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    print("="*70)
    print("  AION - Surgical Partition Patcher ACTIVE")
    print("="*70)
    aion_api.app.run(host="0.0.0.0", port=8100, debug=False)
