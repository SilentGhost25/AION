# scratch/debug_pipeline.py
import sys
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO)

def write_fake_file(path: Path, content: str = "fake content"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path
from acb.acb_pipeline import ACBPipeline

tmp_dir = Path("scratch/test_academic")
subject_code = "BAI404"
subject_dir = tmp_dir / "AIML" / "semester_4" / subject_code

# Clear folder
import shutil
if tmp_dir.exists():
    shutil.rmtree(tmp_dir)

# Write syllabus
write_fake_file(
    subject_dir / "syllabus" / "syllabus.txt",
    "Module 1: Basic AI (5 Hours)\nDFS topic, BFS topic\nCO1: Learn basic search\n"
)

# Write notes
write_fake_file(
    subject_dir / "notes" / "notes_dfs.txt",
    "DFS TOPIC\nDFS topic is defined as depth first search. It uses a stack.\n"
)

pipeline = ACBPipeline(
    subject_code=subject_code,
    academic_root=str(tmp_dir),
    department="AIML",
    semester=4,
)

res = pipeline.run()
print("PIPELINE RESULT:", res)
