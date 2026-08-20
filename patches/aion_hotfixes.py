"""
AION Surgical Partition Patcher
===============================
Only patches the exact partition resolution logic, nothing else.
"""
import os
import sys
import json
import re
import random
import inspect
import functools
import warnings
import threading

warnings.filterwarnings("ignore")

_GLOBAL_DESIRED_SPLIT = [5, 5]
_LOCK = threading.Lock()

def set_active_desired_split(split, exam_type="IAT1"):
    global _GLOBAL_DESIRED_SPLIT
    with _LOCK:
        _GLOBAL_DESIRED_SPLIT = list(split)

def get_active_desired_split():
    with _LOCK:
        return list(_GLOBAL_DESIRED_SPLIT)

# ---------------------------------------------------------------------------
# [1] Marks Parsing
# ---------------------------------------------------------------------------
def parse_desired_split(raw_marks, exam_type="IAT1"):
    is_see = "SEE" in str(exam_type).upper()
    target_sum = 20 if is_see else 10

    if isinstance(raw_marks, (list, tuple)):
        clean = [int(x) for x in raw_marks if str(x).isdigit() and int(x) > 0]
        if clean and sum(clean) == target_sum:
            return clean

    if isinstance(raw_marks, str):
        val = raw_marks.strip().lower()
        if not is_see:
            if val in ("5+5", "5,5", "5 5", "equal", "half"): return [5, 5]
            if val in ("6+4", "6,4", "6 4", "vtu"): return [6, 4]
            if val in ("7+3", "7,3", "7 3"): return [7, 3]
            if val in ("4+3+3", "4,3,3"): return [4, 3, 3]
            if val in ("10", "single"): return [10]
        else:
            if val in ("10+10", "10,10", "equal"): return [10, 10]
            if val in ("8+6+6", "8,6,6"): return [8, 6, 6]
            if val in ("20", "single"): return [20]

        digits = [int(x) for x in re.findall(r"\d+", val)]
        if digits and sum(digits) == target_sum:
            return digits

    return [10, 10] if is_see else [5, 5]

# ---------------------------------------------------------------------------
# [2] Surgical Partition Table Replacement
# ---------------------------------------------------------------------------
def patch_partition_table():
    """Only modifies the PARTITION_TABLE constant and get_partition functions."""
    try:
        import v0_1.main as main_module
        
        desired = get_active_desired_split()
        part_count = len(desired)
        
        # Replace PARTITION_TABLE constant
        if hasattr(main_module, 'PARTITION_TABLE'):
            # Create new table where every count maps to user's choice
            new_table = {
                1: [[10]] if sum(desired) == 10 else [[20]],
                2: [desired] if part_count == 2 else [[6, 4]],
                3: [desired] if part_count == 3 else [[4, 3, 3]]
            }
            main_module.PARTITION_TABLE = new_table
            print(f"[HOTFIX] Replaced PARTITION_TABLE: all counts now map to {desired}")
        
        # Patch get_partition function specifically
        if hasattr(main_module, 'get_partition'):
            orig_get_partition = main_module.get_partition
            
            @functools.wraps(orig_get_partition)
            def new_get_partition(count=2, *args, **kwargs):
                """Always returns user's desired split regardless of count."""
                desired = get_active_desired_split()
                print(f"[HOTFIX] get_partition({count}) → {desired}")
                return desired
            
            main_module.get_partition = new_get_partition
        
        # Patch resolve_partition function if it exists
        if hasattr(main_module, 'resolve_partition'):
            orig_resolve = main_module.resolve_partition
            
            @functools.wraps(orig_resolve)
            def new_resolve(*args, **kwargs):
                desired = get_active_desired_split()
                print(f"[HOTFIX] resolve_partition() → {desired}")
                return desired
            
            main_module.resolve_partition = new_resolve
            
    except Exception as e:
        print(f"[HOTFIX-WARNING] Partition table patch failed: {e}")

# ---------------------------------------------------------------------------
# [3] CleanedDocument Fix
# ---------------------------------------------------------------------------
def patch_cleaned_document_cls(cls):
    if getattr(cls, "_aion_patched", False):
        return
    if hasattr(cls, "model_config"):
        cls.model_config["extra"] = "allow"
    elif hasattr(cls, "Config"):
        cls.Config.extra = "allow"

    orig_init = cls.__init__
    @functools.wraps(orig_init)
    def safe_init(self, *args, **kwargs):
        kwargs.setdefault("removed_line_count", 0)
        kwargs.setdefault("original_line_count", 0)
        try:
            return orig_init(self, *args, **kwargs)
        except TypeError:
            annotations = getattr(cls, "__annotations__", {})
            valid = {k: v for k, v in kwargs.items() if k in annotations}
            valid.setdefault("removed_line_count", 0)
            valid.setdefault("original_line_count", 0)
            orig_init(self, *args, **valid)
            for k, v in kwargs.items():
                if k not in annotations:
                    try:
                        setattr(self, k, v)
                    except:
                        pass
    cls.__init__ = safe_init
    cls._aion_patched = True

# ---------------------------------------------------------------------------
# [4] LLM Wrapper
# ---------------------------------------------------------------------------
def wrap_llm_call(orig_call):
    sig = inspect.signature(orig_call)
    accepted = set(sig.parameters.keys())
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    @functools.wraps(orig_call)
    def safe_call(self, *args, **kwargs):
        if hasattr(self, "temperature"):
            try:
                self.temperature = round(random.uniform(0.72, 0.82), 2)
            except:
                pass
        if not has_kwargs:
            kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        return orig_call(self, *args, **kwargs)
    return safe_call

# ---------------------------------------------------------------------------
# [5] Paper Structure Transform
# ---------------------------------------------------------------------------
def split_text_into_n_parts(text, n=2):
    if n == 1:
        return [text]
    if not text:
        return ["Explain the principles.", "Analyze the parameters."][:n]
    
    # Look for (a), (b), (c) markers
    pattern = r'\s*\([a-c]\)\s*'
    parts = [p.strip() for p in re.split(pattern, text) if p.strip()]
    if len(parts) >= n and all(len(p) > 10 for p in parts[:n]):
        return [p if p.endswith('.') else p + '.' for p in parts[:n]]
    
    # Split on major conjunctions
    conj_pattern = r'(?:,\s*(?:and|as well as)\s+(?:explain|justify|analyze|calculate|compare)\s+)'
    matches = list(re.finditer(conj_pattern, text, re.I))
    if len(matches) >= n - 1:
        parts = []
        last = 0
        for m in matches[:n-1]:
            parts.append(text[last:m.start()].strip().rstrip(','))
            last = m.end()
        parts.append(text[last:].strip())
        if all(len(p) > 15 for p in parts):
            return [p if p.endswith('.') else p + '.' for p in parts]
    
    # Fallback
    return [text, "Justify the underlying principles and parameters."][:n]

def transform_paper_structure(paper, desired_split=None):
    if not isinstance(paper, dict):
        return paper
    if desired_split is None:
        desired_split = get_active_desired_split()

    target_count = len(desired_split)
    target_sum = sum(desired_split)

    modules = paper.get("modules", [])
    if not modules and "paper" in paper:
        modules = paper["paper"].get("modules", [])

    for mod in modules:
        if not isinstance(mod, dict):
            continue
        for q in mod.get("questions", []):
            if not isinstance(q, dict):
                continue
            subs = q.get("sub_questions", [])

            if len(subs) < target_count:
                orig = subs[0].get("text", q.get("text", "")) if subs else q.get("text", "")
                parts = split_text_into_n_parts(orig, target_count)
                q["sub_questions"] = []
                for i in range(target_count):
                    q["sub_questions"].append({
                        "label": f"({chr(97+i)})",
                        "letter": chr(97+i),
                        "text": parts[i],
                        "marks": desired_split[i],
                        "co": subs[i].get("co", "CO1") if i < len(subs) else ("CO1" if i == 0 else "CO2"),
                        "bloom": subs[i].get("bloom", "L3") if i < len(subs) else ("L3" if i == 0 else "L2")
                    })
            else:
                for i, sq in enumerate(subs[:target_count]):
                    sq["marks"] = desired_split[i]
                    sq["label"] = f"({chr(97+i)})"
                    sq["letter"] = chr(97+i)
                q["sub_questions"] = subs[:target_count]

            q["marks"] = target_sum
            q["total_marks"] = target_sum

    return paper
