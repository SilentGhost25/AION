import re

api_file = "/home/AIML1/AIQ/AION/aion_api.py"

with open(api_file, "r") as f:
    lines = f.readlines()

# Step 1: Remove ALL remnants of previous broken patches
skip_keywords = ["_exam", "_is_see", "_target", "_digits", "_split", 
                 "set_user_marks_split", "marks_distribution", 
                 "User marks distribution", "MARKS DISTRIBUTION"]

clean_lines = []
for line in lines:
    if any(kw in line for kw in skip_keywords):
        continue
    clean_lines.append(line)

# Step 2: Find 'sub_question_count' extraction and insert clean block
final_lines = []
inserted = False

for line in clean_lines:
    final_lines.append(line)
    
    if not inserted and 'sub_question_count' in line and 'data.get' in line:
        indent = len(line) - len(line.lstrip())
        indent_str = ' ' * indent
        
        block = f"""
{indent_str}# --- USER MARKS SPLIT INJECTION ---
{indent_str}try:
{indent_str}    _marks_dist = data.get("marks_distribution") or data.get("marks_split") or None
{indent_str}    if _marks_dist:
{indent_str}        _is_see_ = "SEE" in str(exam_type).upper()
{indent_str}        _target_ = 20 if _is_see_ else 10
{indent_str}        _digits_ = [int(x) for x in __import__("re").findall(r"\\d+", str(_marks_dist))]
{indent_str}        _split_ = [5, 5] if not _is_see_ else [10, 10]
{indent_str}        if _digits_ and sum(_digits_) == _target_: _split_ = _digits_
{indent_str}        elif "6" in str(_marks_dist) and not _is_see_: _split_ = [6, 4]
{indent_str}        elif "7" in str(_marks_dist) and not _is_see_: _split_ = [7, 3]
{indent_str}        elif "10" in str(_marks_dist) and _is_see_: _split_ = [10, 10]
{indent_str}        from v0_1.main import set_user_marks_split
{indent_str}        set_user_marks_split(_split_)
{indent_str}        sub_question_count = len(_split_)
{indent_str}        print(f"[API] User marks distribution: {{_marks_dist}} -> {{_split_}}")
{indent_str}except Exception as _me:
{indent_str}    print(f"[API] Marks dist error: {{_me}}")
{indent_str}# --- END INJECTION ---
"""
        final_lines.append(block)
        inserted = True

with open(api_file, "w") as f:
    f.writelines(final_lines)

print("[FIX] ✓ aion_api.py fully sanitized and patched")
