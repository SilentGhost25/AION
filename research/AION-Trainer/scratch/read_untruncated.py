# scratch/read_untruncated.py
import json

transcript_path = r"C:\Users\Tarun J\.gemini\antigravity-ide\brain\9cb2f04d-f10f-4294-9988-645b89242c9f\.system_generated\logs\transcript_full.jsonl"

with open(transcript_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get("type") == "USER_INPUT":
                last_user_input = data
        except Exception:
            pass

if "last_user_input" in locals():
    content = last_user_input.get("content", "")
    print(f"Length of content: {len(content)}")
    # Write to a file in scratch
    with open("scratch/full_user_request.txt", "w", encoding="utf-8") as out:
        out.write(content)
    print("Successfully wrote full content to scratch/full_user_request.txt")
else:
    print("Could not find USER_INPUT")

