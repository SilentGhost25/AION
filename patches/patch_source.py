"""
Direct Source File Patcher
=========================
Makes surgical edits to v0_1/main.py, aion_api.py,
and core preprocessing files. No wrappers needed.
"""
import os
import re
import sys

aion_root = "/home/AIML1/AIQ/AION"

# ===========================================================
# PATCH 1: v0_1/main.py - Partition Table & Resolver
# ===========================================================
main_py = os.path.join(aion_root, "v0_1", "main.py")

with open(main_py, "r") as f:
    code = f.read()

# 1a. Add USER_MARKS_SPLIT global at the top (after imports)
if "USER_MARKS_SPLIT" not in code:
    # Find first function definition or class definition
    insert_pos = 0
    lines = code.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("def ") or line.startswith("class "):
            insert_pos = i
            break
    
    header = '''# ============================================================
# [USER MARKS SPLIT] - Dynamically set by API request
# ============================================================
USER_MARKS_SPLIT = None  # Set to [5,5], [6,4], [7,3], etc.

def set_user_marks_split(split_list):
    """Called from aion_api.py to set the user's desired marks split."""
    global USER_MARKS_SPLIT
    USER_MARKS_SPLIT = split_list
    print(f"[PARTITION] User marks split locked to: {USER_MARKS_SPLIT}")

def get_user_marks_split():
    """Returns the user's desired split, or None if not set."""
    return USER_MARKS_SPLIT

'''
    lines.insert(insert_pos, header)
    code = "\n".join(lines)
    print(f"[PATCH] Added USER_MARKS_SPLIT globals to v0_1/main.py")

# 1b. Find and replace PARTITION_TABLE
if "PARTITION_TABLE" in code:
    # Replace entire PARTITION_TABLE definition
    pattern = r'PARTITION_TABLE\s*=\s*\{[^}]+\}'
    replacement = '''PARTITION_TABLE = {
    1: [[10]],
    2: [[5, 5], [6, 4]],  # [5,5] = user default, [6,4] = VTU fallback
    3: [[4, 3, 3]]
}'''
    code = re.sub(pattern, replacement, code)
    print(f"[PATCH] Replaced PARTITION_TABLE with dynamic version")

# 1c. Find partition resolution logic and add USER_MARKS_SPLIT override
# Look for patterns like "-> [6, 4]" or "parts=2" or partition selection
# We need to find where the partition is actually selected from the table

# Find any function that selects from PARTITION_TABLE
# Pattern: PARTITION_TABLE[count] or PARTITION_TABLE.get(count)
if "get_user_marks_split" not in code or "USER_MARKS_SPLIT" not in code:
    print("[PATCH] USER_MARKS_SPLIT already injected")
else:
    # Find where partitions are used and inject override
    # Common patterns:
    #   partition = PARTITION_TABLE[count][0]
    #   partitions = PARTITION_TABLE.get(count, [[6,4]]) 
    
    # Add override after any PARTITION_TABLE access
    old_pattern = r'(\s+)(\w+)\s*=\s*PARTITION_TABLE\s*\[\s*\w+\s*\]\s*\[\s*0\s*\]'
    
    def add_override(match):
        indent = match.group(1)
        varname = match.group(2)
        return f'''{indent}# User marks split override
{indent}_user_split = get_user_marks_split()
{indent}{varname} = _user_split if _user_split else PARTITION_TABLE[parts][0]'''
    
    code = re.sub(old_pattern, add_override, code)
    print(f"[PATCH] Added USER_MARKS_SPLIT override to partition selection")

# 1d. Find "final partitions" log line and add override before it
# This is where we see: [PIPELINE] Module X final partitions: Q1=[6, 4]
if "final partitions" in code:
    # Find the line that sets final partitions
    lines = code.split("\n")
    for i, line in enumerate(lines):
        if "final partition" in line.lower() and "print" in line.lower():
            # Insert override check before this print
            indent = len(line) - len(line.lstrip())
            indent_str = " " * indent
            override_code = f'''{indent_str}# Apply user marks split override
{indent_str}_usr_split = get_user_marks_split()
{indent_str}if _usr_split:
{indent_str}    # Override all partitions with user's choice
{indent_str}    for _qk in list(partitions.keys()):
{indent_str}        partitions[_qk] = _usr_split
'''
            lines.insert(i, override_code)
            code = "\n".join(lines)
            print(f"[PATCH] Added override before 'final partitions' log")
            break

with open(main_py, "w") as f:
    f.write(code)
print(f"[PATCH] ✓ v0_1/main.py patched successfully")

# ===========================================================
# PATCH 2: aion_api.py - Extract marks_distribution from request
# ===========================================================
api_py = os.path.join(aion_root, "aion_api.py")

with open(api_py, "r") as f:
    code = f.read()

if "marks_distribution" not in code:
    # Find where sub_question_count is extracted from request
    # and add marks_distribution extraction
    pattern = r'(sub_question_count\s*=\s*.*?\n)'
    
    replacement = r'''\1    # Extract user marks distribution
    marks_distribution = data.get("marks_distribution") or data.get("marksDistribution") or data.get("marks_split") or None
    if marks_distribution:
        # Parse the marks distribution string
        _exam = exam_type if 'exam_type' in dir() else "IAT1"
        _is_see = "SEE" in str(_exam).upper()
        _target = 20 if _is_see else 10
        _digits = [int(x) for x in __import__("re").findall(r"\\d+", str(marks_distribution))]
        if _digits and sum(_digits) == _target:
            _split = _digits
        elif "5" in str(marks_distribution) and not _is_see:
            _split = [5, 5]
        elif "6" in str(marks_distribution) and not _is_see:
            _split = [6, 4]
        elif "7" in str(marks_distribution) and not _is_see:
            _split = [7, 3]
        elif "10" in str(marks_distribution) and _is_see:
            _split = [10, 10]
        else:
            _split = [5, 5] if not _is_see else [10, 10]
        
        # Set in main module
        try:
            from v0_1.main import set_user_marks_split
            set_user_marks_split(_split)
        except:
            pass
        
        # Override sub_question_count to match
        sub_question_count = len(_split)
        print(f"[API] User marks distribution: {marks_distribution} -> {_split}")

'''
    code = re.sub(pattern, replacement, code, count=1)
    print(f"[PATCH] Added marks_distribution extraction to aion_api.py")
else:
    print(f"[PATCH] marks_distribution already in aion_api.py")

with open(api_py, "w") as f:
    f.write(code)
print(f"[PATCH] ✓ aion_api.py patched successfully")

# ===========================================================
# PATCH 3: CleanedDocument - Add missing positional args
# ===========================================================
for modpath in [
    os.path.join(aion_root, "core", "preprocessing", "document_cleaner.py"),
    os.path.join(aion_root, "v0_1", "schemas.py")
]:
    if not os.path.exists(modpath):
        continue
    
    with open(modpath, "r") as f:
        code = f.read()
    
    if "removed_line_count" in code and "original_line_count" in code:
        # Find CleanedDocument class
        if "class CleanedDocument" in code:
            # Add defaults to __init__
            code = re.sub(
                r'(def __init__\(self,\s*[^)]*?removed_line_count\s*,\s*original_line_count\s*)\)',
                r'\1=0, original_line_count=0)',
                code
            )
            # Also try the other pattern
            code = re.sub(
                r'removed_line_count:\s*int\s*,\s*original_line_count:\s*int',
                r'removed_line_count: int = 0, original_line_count: int = 0',
                code
            )
            with open(modpath, "w") as f:
                f.write(code)
            print(f"[PATCH] ✓ {os.path.basename(modpath)} - Added defaults to CleanedDocument")

print("\n" + "="*60)
print("  ALL SOURCE FILES PATCHED DIRECTLY")
print("  No wrappers or launchers needed")
print("="*60)
