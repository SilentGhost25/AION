#!/usr/bin/env python3
"""
AION Server Bridge
Called by Express via child_process.spawn().
Reads config from environment variables.
Prints JSON result as last stdout line.
"""

import os
import sys
import json
from pathlib import Path

# Setup
os.chdir(Path(__file__).parent)
sys.path.insert(0, ".")
os.environ.setdefault("AION_MODEL", "qwen2.5:3b")

def main():
    files_env   = os.environ.get("AION_FILES", "[]")
    try:
        files = json.loads(files_env)
    except Exception as e:
        print(f"Error parsing AION_FILES environment variable: {e}", file=sys.stderr)
        sys.exit(1)

    subject     = os.environ.get("AION_SUBJECT", "Unknown")
    exam_type   = os.environ.get("AION_EXAM",    "see")
    mode        = os.environ.get("AION_MODE",    "turbo")
    max_n       = int(os.environ.get("AION_N",   "10"))

    if not files:
        print(json.dumps({"error": "No files provided"}), file=sys.stderr)
        sys.exit(1)

    # Use first file as primary generation source
    file_path = files[0]

    try:
        from v0_1.main import run_pipeline
    except ImportError as e:
        print(f"Failed to import v0_1.main: {e}. PYTHONPATH={sys.path}", file=sys.stderr)
        sys.exit(1)

    try:
        # run_pipeline returns (output_paper, [])
        paper, _ = run_pipeline(
            file_path,
            max_concepts = max_n,
            mode         = mode,
            exam_type    = exam_type,
        )
    except Exception as e:
        import traceback
        print(f"Pipeline generation failed: {e}\n{traceback.format_exc()}", file=sys.stderr)
        sys.exit(1)

    # Calculate marks dynamically from generated subquestions
    total_marks = 0
    for mod in paper:
        for mq in mod.get("questions", []):
            for sq in mq.get("sub_questions", []):
                total_marks += sq.get("marks", 0)

    # Serialize result
    result = {
        "id":          __import__("uuid").uuid4().hex,
        "subject":     subject,
        "examType":    exam_type,
        "mode":        mode,
        "modules":     _serialize(paper),
        "generatedAt": __import__("datetime").datetime.now().isoformat(),
        "totalMarks":  total_marks,
    }

    # Print JSON as last line (Express reads this)
    print(json.dumps(result))


def _serialize(modules):
    """Convert module dicts to JSON-safe format."""
    out = []
    for mod in modules:
        questions = []
        for mq in mod.get("questions", []):
            questions.append({
                "mqIndex":     mq.get("mq_index"),
                "bloomLevel":  mq.get("bloom_level"),
                "bloomName":   mq.get("bloom_name"),
                "totalMarks":  mq.get("total_marks"),
                "subQuestions": [
                    {
                        "letter": sq.get("letter"),
                        "text":   sq.get("text"),
                        "marks":  sq.get("marks"),
                    }
                    for sq in mq.get("sub_questions", [])
                ],
            })
        out.append({
            "moduleIndex": mod.get("module_index"),
            "moduleTitle": mod.get("module_title"),
            "questions":   questions,
        })
    return out


if __name__ == "__main__":
    main()
