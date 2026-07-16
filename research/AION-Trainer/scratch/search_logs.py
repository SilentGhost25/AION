# scratch/search_logs.py
import json
import os

target_str = "class DocumentClassifier"
import json
import os

target_str = "course_preview_builder"
brain_dir = r"C:\Users\Tarun J\.gemini\antigravity-ide\brain"

found = False
for folder in os.listdir(brain_dir):
    logs_dir = os.path.join(brain_dir, folder, ".system_generated", "logs")
    full_path = os.path.join(logs_dir, "transcript_full.jsonl")
    if not os.path.exists(full_path):
        continue
    with open(full_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if target_str in line:
                print(f"Found in folder: {folder}, line: {idx}")
                found = True
                break



if not found:
    print("Could not find any matching log with length > 10k")
