"""
AION Dynamic Marks Partition Engine
"""
from typing import List, Optional, Any, Union
import re
import threading

_lock = threading.Lock()
_active_split: Optional[List[int]] = None
_active_exam: str = "IAT1"


def set_user_split(split: Optional[List[int]], exam_type: str = "IAT1") -> None:
    global _active_split, _active_exam
    with _lock:
        _active_split = list(split) if split else None
        _active_exam = str(exam_type).upper()
        if _active_split:
            print(f"[MARKS-PARTITIONER] Active split locked to: {_active_split} for {_active_exam}", flush=True)


def get_user_split() -> Optional[List[int]]:
    with _lock:
        return list(_active_split) if _active_split else None


def parse_marks(
    raw: Any, 
    exam_type: str = "IAT1", 
    sub_question_count: Optional[Union[int, List[int]]] = None
) -> Optional[List[int]]:
    total = 20 if "SEE" in str(exam_type).upper() else 10

    if isinstance(raw, (list, tuple)) and raw:
        clean = [int(x) for x in raw if str(x).isdigit() and int(x) > 0]
        if clean and sum(clean) == total:
            return clean
        if len(clean) == 1 and clean[0] == total:
            return clean

    if raw and isinstance(raw, str):
        val = raw.strip().lower()
        val = re.sub(r'[m\s_,\-\+]*marks?', '', val)
        val = val.rstrip('m').strip()

        presets_iat = {
            "5+5": [5, 5], "5,5": [5, 5], "5 5": [5, 5], "equal": [5, 5], "half": [5, 5],
            "6+4": [6, 4], "6,4": [6, 4], "6 4": [6, 4], "vtu": [6, 4], "standard": [6, 4],
            "7+3": [7, 3], "7,3": [7, 3], "7 3": [7, 3],
            "8+2": [8, 2], "8,2": [8, 2],
            "9+1": [9, 1], "9,1": [9, 1],
            "4+3+3": [4, 3, 3], "4,3,3": [4, 3, 3], "4 3 3": [4, 3, 3],
            "4+4+2": [4, 4, 2], "4,4,2": [4, 4, 2],
            "3+3+2+2": [3, 3, 2, 2],
            "10": [10], "single": [10], "full": [10], "1": [10]
        }

        presets_see = {
            "10+10": [10, 10], "10,10": [10, 10], "equal": [10, 10], "half": [10, 10],
            "12+8": [12, 8], "12,8": [12, 8],
            "14+6": [14, 6], "14,6": [14, 6],
            "8+6+6": [8, 6, 6], "8,6,6": [8, 6, 6],
            "7+7+6": [7, 7, 6], "7,7,6": [7, 7, 6],
            "6+6+8": [6, 6, 8], "6,6,8": [6, 6, 8],
            "20": [20], "single": [20], "full": [20], "1": [20]
        }

        presets = presets_see if total == 20 else presets_iat
        if val in presets:
            return presets[val]

        digits = [int(x) for x in re.findall(r"\d+", val)]
        if digits and sum(digits) == total:
            return digits

    sq_c = None
    if isinstance(sub_question_count, int) and sub_question_count > 0:
        sq_c = sub_question_count
    elif isinstance(sub_question_count, list) and sub_question_count:
        sq_c = sub_question_count[0]
    elif raw and str(raw).isdigit():
        sq_c = int(raw)

    if sq_c:
        if total == 10:
            if sq_c == 1: return [10]
            if sq_c == 2: return [5, 5]
            if sq_c == 3: return [4, 3, 3]
            if sq_c == 4: return [3, 3, 2, 2]
        else:
            if sq_c == 1: return [20]
            if sq_c == 2: return [10, 10]
            if sq_c == 3: return [8, 6, 6]
            if sq_c == 4: return [5, 5, 5, 5]

    return None


def resolve_partition(parts: int = 2, total_marks: int = 10, user_pref: Optional[str] = None) -> List[int]:
    active = get_user_split()
    if active and sum(active) == total_marks:
        return list(active)

    if user_pref:
        parsed = parse_marks(user_pref, "SEE" if total_marks == 20 else "IAT1")
        if parsed and sum(parsed) == total_marks:
            return parsed

    if parts <= 1:
        return [total_marks]

    base = total_marks // parts
    remainder = total_marks % parts
    return [base + (1 if i < remainder else 0) for i in range(parts)]


parse_user_split = parse_marks
set_global_user_split = set_user_split
get_global_user_split = get_user_split
resolve_module_partitions = resolve_partition
