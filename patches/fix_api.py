import re

api_file = "/home/AIML1/AIQ/AION/aion_api.py"

with open(api_file, "r") as f:
    lines = f.readlines()

# Find and remove the broken marks_distribution block
# Then insert it correctly in the right place
new_lines = []
skip_until_blank = False
found_broken = False
insert_done = False

i = 0
while i < len(lines):
    line = lines[i]
    
    # Detect the broken marks_distribution block
    if 'marks_distribution = data.get("marks_distribution")' in line and not line.strip().startswith('#'):
        found_broken = True
        # Skip this entire block until we hit a blank line or a line that looks like original code
        i += 1
        while i < len(lines):
            next_line = lines[i]
            # Check if we've exited the broken block
            if next_line.strip() == '' or next_line.strip().startswith('except') or next_line.strip().startswith('finally'):
                # We've reached the end of the broken block
                break
            if 'sub_question_count' in next_line and 'len(_split)' in next_line:
                i += 1
                continue
            if '[API] User marks distribution' in next_line:
                i += 1
                continue
            if 'marks_distribution' in next_line or '_exam' in next_line or '_is_see' in next_line or '_target' in next_line or '_digits' in next_line or '_split' in next_line or 'set_user_marks_split' in next_line:
                i += 1
                continue
            break
        continue
    
    new_lines.append(line)
    i += 1

# Now find where sub_question_count is set and add the marks_distribution logic properly
final_lines = []
for i, line in enumerate(new_lines):
    final_lines.append(line)
    
    # Find the line that sets sub_question_count from request data
    if not insert_done and 'sub_question_count' in line and 'data.get' in line:
        # Get the indentation of this line
        indent = len(line) - len(line.lstrip())
        indent_str = ' ' * indent
        
        # Get the surrounding try block indentation
        # Look backwards for 'try:'
        parent_indent = indent - 4
        parent_str = ' ' * parent_indent
        
        # Insert marks_distribution logic right after sub_question_count
        insert_block = f'''{indent_str}# ---- USER MARKS DISTRIBUTION ----
{indent_str}marks_distribution = data.get("marks_distribution") or data.get("marksDistribution") or data.get("marks_split") or None
{indent_str}if marks_distribution:
{indent_str}    try:
{indent_str}        _is_see = "SEE" in str(exam_type).upper() if 'exam_type' in dir() else False
{indent_str}        _target = 20 if _is_see else 10
{indent_str}        _digits = [int(x) for x in __import__("re").findall(r"\\d+", str(marks_distribution))]
{indent_str}        if _digits and sum(_digits) == _target:
{indent_str}            _split = _digits
{indent_str}        elif "5" in str(marks_distribution) and not _is_see:
{indent_str}            _split = [5, 5]
{indent_str}        elif "6" in str(marks_distribution) and not _is_see:
{indent_str}            _split = [6, 4]
{indent_str}        elif "7" in str(marks_distribution) and not _is_see:
{indent_str}            _split = [7, 3]
{indent_str}        elif "10" in str(marks_distribution) and _is_see:
{indent_str}            _split = [10, 10]
{indent_str}        else:
{indent_str}            _split = [5, 5] if not _is_see else [10, 10]
{indent_str}        from v0_1.main import set_user_marks_split
{indent_str}        set_user_marks_split(_split)
{indent_str}        sub_question_count = len(_split)
{indent_str}        print(f"[API] User marks distribution: {{marks_distribution}} -> {{_split}}")
{indent_str}    except Exception as _me:
{indent_str}        print(f"[API] Marks distribution parse error: {{_me}}")
{indent_str}# ---- END USER MARKS DISTRIBUTION ----
'''
        final_lines.append(insert_block)
        insert_done = True

with open(api_file, "w") as f:
    f.writelines(final_lines)

print(f"[FIX] ✓ aion_api.py syntax fixed")
print(f"[FIX] Removed broken block: {found_broken}")
print(f"[FIX] Inserted clean block: {insert_done}")
