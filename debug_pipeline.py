#!/usr/bin/env python3
"""
AION Pipeline Debug Script
Run: python debug_pipeline.py
"""

import sys
import traceback
from pathlib import Path

# Setup root path
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

print("=" * 50)
print("AION Pipeline Debug Test")
print("=" * 50)

# Step 1: Import pipeline
print("\n[1] Importing run_pipeline...")
try:
    from v0_1.main import run_pipeline
    print("    OK: Import succeeded")
except Exception as e:
    print(f"    FAIL: Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 2: Find test file
print("\n[2] Looking for test file...")
test_file = None
uploads = ROOT / "workspace" / "uploads"
docx_list = list(uploads.glob("*.docx")) if uploads.exists() else []
if docx_list:
    test_file = str(docx_list[0])
    print(f"    OK: Found test DOCX file: {test_file}")
elif uploads.exists():
    files = list(uploads.iterdir())
    if files:
        test_file = str(files[0])
        print(f"    OK: Found test file: {test_file}")
    else:
        print("    FAIL: No files found in workspace/uploads.")
        print("    Please upload a file via the UI first.")
        sys.exit(1)
else:
    print("    FAIL: Directory workspace/uploads does not exist.")
    sys.exit(1)

# Step 3: Run pipeline
print("\n[3] Running pipeline test...")
print("    Processing (this may take a few moments)...")
print("-" * 50)

try:
    result = run_pipeline(
        test_file,
        max_concepts = 3,
        mode         = "turbo",
        exam_type    = "see",
    )
    print("-" * 50)
    print(f"\n    OK: Pipeline completed successfully!")
    print(f"    Result type: {type(result)}")
    print(f"    Result preview: {str(result)[:500]}")
except Exception as e:
    print("-" * 50)
    print(f"\n    FAIL: Pipeline failed: {e}")
    traceback.print_exc()
