"""
AION Dynamic Marks Partition Engine
"""
from typing import List, Optional
import re

_ACTIVE_USER_SPLIT: Optional[List[int]] = None

def set_global_user_split(split: Optional[List[int]]):
    global _ACTIVE_USER_SPLIT
    _ACTIVE_USER_SPLIT = list(split) if split else None

def get_global_user_split() -> Optional[List[int]]:
    global _ACTIVE_USER_SPLIT
    return _ACTIVE_USER_SPLIT

def parse_user_split(raw_input: Optional[str], total_marks: int = 10) -> Optional[List[int]]:
    if not raw_input:
        return None
    val = str(raw_input).strip().lower()
    if val in ("5+5", "5,5", "5 5", "5/5", "equal", "half", "even"):
        return [5, 5] if total_marks == 10 else [10, 10]
    if val in ("6+4", "6,4", "6 4", "6/4", "vtu"):
        return [6, 4]
    if val in ("7+3", "7,3", "7 3"):
        return [7, 3]
    if val in ("8+2", "8,2"):
        return [8, 2]
    if val in ("4+3+3", "4,3,3"):
        return [4, 3, 3]
    if val in ("10", "single", "full"):
        return [10]
    if val in ("10+10", "10,10"):
        return [10, 10]
    if val in ("8+6+6", "8,6,6"):
        return [8, 6, 6]
    digits = [int(x) for x in re.findall(r"\d+", val)]
    if digits and sum(digits) == total_marks:
        return digits
    return None

def resolve_partition(parts: int = 2, total_marks: int = 10, user_pref: Optional[str] = None) -> List[int]:
    active = get_global_user_split()
    if active and sum(active) == total_marks:
        return list(active)
    if user_pref:
        parsed = parse_user_split(user_pref, total_marks)
        if parsed:
            return parsed
    if parts <= 1:
        return [total_marks]
    if parts == 2:
        half = total_marks // 2
        return [half, total_marks - half]
    if parts == 3:
        return [4, 3, 3] if total_marks == 10 else [8, 6, 6]
    base = total_marks // parts
    rem = total_marks % parts
    return [base + (1 if i < rem else 0) for i in range(parts)]
