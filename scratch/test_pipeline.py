import sys
import os
from pathlib import Path

# Add project root to path
ROOT = Path("c:/Users/Tarun J/OneDrive/Desktop/AION")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from v0_1.main import run_pipeline

file_path = "workspace/uploads/074c9920-6b0.pdf"
print(f"Testing run_pipeline on {file_path} ...")
try:
    paper, rejected = run_pipeline(
        file_path      = file_path,
        max_concepts   = 5,
        mode           = "turbo",
        exam_type      = "see",
        difficulty     = "mixed",
        include_visual = True,
    )
    print("SUCCESS! Generated paper modules count:", len(paper))
except Exception as e:
    import traceback
    print("FAILED with exception:", e)
    traceback.print_exc()
