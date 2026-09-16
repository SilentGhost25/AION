import urllib.request
import json
import time
import sys
import os
import fitz  # PyMuPDF

print("=== STARTING 2-FILE END-TO-END VERIFICATION TEST (REAL PDFS) ===")

f1_content = """Module 1: Keplerian Orbit Mechanics and Satellite Subsystems
A communication satellite consists of a space segment and ground segment. 
Keplerian orbital parameters define the satellite trajectory including semi-major axis, 
eccentricity, inclination, argument of perigee, and right ascension of ascending node. 
Geostationary earth orbits operate at altitude of 35786 kilometers with orbital period of 24 hours.
The satellite payload comprises transponders, multiplexers, and high-gain parabolic antennas.
Spacecraft telemetry tracking and command subsystems maintain proper satellite attitude and orbital station-keeping."""

f2_content = """Module 2: Satellite Link Budget Analysis and Atmospheric Loss
The satellite communication link budget calculates received carrier power using Friis transmission equation.
Equivalent Isotropically Radiated Power EIRP represents the product of transmit power and antenna gain.
Path loss and rain attenuation degrade downlink signal strength at higher microwave frequencies.
Carrier to noise ratio CNR and system noise temperature determine receiver sensitivity and bit error rate.
Link margin calculations account for atmospheric absorption, antenna pointing errors, and ionospheric scintillation."""

# Create real PDF 1
doc1 = fitz.open()
page1 = doc1.new_page()
page1.insert_text((50, 72), f1_content, fontsize=12)
pdf1_path = "scratch_module1.pdf"
doc1.save(pdf1_path)
doc1.close()

# Create real PDF 2
doc2 = fitz.open()
page2 = doc2.new_page()
page2.insert_text((50, 72), f2_content, fontsize=12)
pdf2_path = "scratch_module2.pdf"
doc2.save(pdf2_path)
doc2.close()

print(f"[0] Created real PDF files: {pdf1_path} ({os.path.getsize(pdf1_path)} bytes), {pdf2_path} ({os.path.getsize(pdf2_path)} bytes)")

print("[1] Uploading PDF 1 (Module 1: Keplerian Orbit Mechanics)...")

def upload_file(path, filename, subject="Satellite Communication"):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    with open(path, "rb") as f:
        file_bytes = f.read()
    
    parts = []
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="subject"\r\n\r\n{subject}\r\n'.encode('utf-8'))
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="category"\r\n\r\nnotes\r\n'.encode('utf-8'))
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: application/pdf\r\n\r\n'.encode('utf-8') + file_bytes + b'\r\n')
    parts.append(f'--{boundary}--\r\n'.encode('utf-8'))
    body = b"".join(parts)
    
    req = urllib.request.Request(
        "http://localhost:8100/api/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

res1 = upload_file(pdf1_path, "module1_orbits.pdf")
id1 = res1.get("document_id") or res1.get("id")
print(f" -> Upload 1 OK: doc_id={id1}, status={res1.get('status')}")

print("[2] Uploading PDF 2 (Module 2: Link Budget & EIRP)...")
res2 = upload_file(pdf2_path, "module2_linkbudget.pdf")
id2 = res2.get("document_id") or res2.get("id")
print(f" -> Upload 2 OK: doc_id={id2}, status={res2.get('status')}")

# Call /api/generate/stream with 2-file payload matching frontend
payload = {
    "file_id": id1,
    "fileId": id1,
    "file_ids": [id1, id2],
    "fileIds": [id1, id2],
    "module_files": {"1": id1, "2": id2},
    "moduleFiles": {"1": id1, "2": id2},
    "subject": "Satellite Communication",
    "department": "Electronics and Communication Engineering",
    "semester": 6,
    "exam_type": "IA",
    "difficulty": "mixed",
    "model": "qwen2.5:1.5b-instruct",
    "sub_question_counts": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    "mark_splits": [[6, 4], [6, 4], [6, 4], [6, 4], [6, 4], [6, 4], [6, 4], [6, 4], [6, 4], [6, 4]],
}

req_data = json.dumps(payload).encode("utf-8")
gen_req = urllib.request.Request(
    "http://localhost:8100/api/generate/stream",
    data=req_data,
    headers={"Content-Type": "application/json"}
)

print("[3] Streaming generation from /api/generate/stream...")
paper_data = None
start_t = time.time()

with urllib.request.urlopen(gen_req, timeout=300) as resp:
    for line in resp:
        line_str = line.decode("utf-8", errors="replace").strip()
        if line_str.startswith("data:"):
            data_body = line_str[5:].strip()
            try:
                parsed = json.loads(data_body)
                if "stage" in parsed and "message" in parsed:
                    print(f"  [STAGE] {parsed['stage']}: {parsed['message']}")
                if parsed.get("status") == "SUCCESS" and "paper" in parsed:
                    paper_data = parsed["paper"]
                    print("\n[SUCCESS] Final paper received via SSE stream!")
                elif "paper" in parsed and isinstance(parsed["paper"], dict) and "modules" in parsed["paper"]:
                    paper_data = parsed["paper"]
                elif "error" in parsed:
                    print(f"\n[PIPELINE ERROR]: {parsed['error']}")
            except Exception:
                pass

dur = time.time() - start_t
print(f"Generation finished in {dur:.1f}s.")

# Cleanup temp files
for p in [pdf1_path, pdf2_path]:
    if os.path.exists(p):
        os.remove(p)

if not paper_data:
    print("\n[FAIL] No paper was returned by the stream!")
    sys.exit(1)

# Inspect generated questions
print("\n" + "=" * 70)
print("=== VERIFYING GENERATED PAPER QUESTIONS ACROSS MODULES ===")
print("=" * 70)

modules = paper_data.get("modules", [])
print(f"Total modules in paper: {len(modules)}")

all_questions_text = []

for mod in modules:
    mod_idx = mod.get("module_index") or mod.get("moduleIndex")
    mod_title = mod.get("module_title") or mod.get("title") or f"Module {mod_idx}"
    print(f"\n--- Module {mod_idx}: {mod_title} ---")
    for q in mod.get("questions", []):
        q_no = q.get("question_no") or q.get("mq_index") or q.get("questionNumber")
        print(f"  Main Question {q_no}:")
        subs = q.get("sub_questions") or q.get("subQuestions") or []
        for s in subs:
            let = s.get("letter") or s.get("label") or "a"
            txt = s.get("text", "")
            m = s.get("marks", 0)
            print(f"    ({let}) [{m}M] {txt[:120]}...")
            all_questions_text.append(txt)

combined_text = " ".join(all_questions_text).lower()

# Verification check: Content from File 1 (Kepler, orbit, geostationary, perigee, payload)
file1_signals = ["orbit", "kepler", "geostationary", "perigee", "payload", "transponder", "satellite"]
f1_found = [sig for sig in file1_signals if sig in combined_text]

# Verification check: Content from File 2 (link budget, friis, eirp, attenuation, carrier, noise)
file2_signals = ["link", "budget", "friis", "eirp", "attenuation", "rain", "carrier", "noise"]
f2_found = [sig for sig in file2_signals if sig in combined_text]

print("\n" + "=" * 70)
print("=== END-TO-END VALIDATION SUMMARY ===")
print("=" * 70)
print(f"File 1 content signals found: {f1_found}")
print(f"File 2 content signals found: {f2_found}")

if f1_found and f2_found:
    print("\n>>> TEST PASSED: Content from BOTH uploaded files was successfully extracted, synthesized, and generated into questions! <<<")
else:
    print(f"\n>>> TEST FAILED: Missing signals! File 1 found: {bool(f1_found)}, File 2 found: {bool(f2_found)} <<<")
    sys.exit(1)
