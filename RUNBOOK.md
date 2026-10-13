# Server Runbook — First 30 Minutes After Login

This runbook guides server-side verification and deployment for AION v3.
Follow these steps strictly in order upon logging into the GPU host (`AIML1@testing`).

---

## 0. Sync & Workspace Verification
```bash
cd ~/AIQ/AION
git fetch origin v3
git checkout v3
git pull origin v3
git log -3 --oneline
```
Ensure you are on the latest `v3` branch commit.

---

## 1. Verify / Fix Model Environment (`AION_MODEL`)
Ensure the local LLM model matches what vLLM serves (`Qwen/Qwen2.5-14B-Instruct-AWQ`):
```bash
# Check if any legacy 'qwen2.5:14b' remains
grep -rn "qwen2.5:14b" ~/.bashrc ~/AIQ/AION/.env.server ~/AIQ/AION/aiq 2>/dev/null

# Ensure AION_MODEL is exported correctly in environment
export AION_MODEL="Qwen/Qwen2.5-14B-Instruct-AWQ"
export AION_BACKEND="vllm"
export VLLM_PORT=8000
```
*(Note: `./aiq` now has universal auto-alignment that dynamically selects `Qwen/Qwen2.5-14B-Instruct-AWQ` if port 8000 responds to vLLM health checks).*

---

## 2. Launch / Restart Backend & Services
Stop any stale processes and launch the backend:
```bash
cd ~/AIQ/AION
./aiq stop 2>/dev/null; sleep 2

# Start backend with tee logging
./aiq backend 2>&1 | tee /tmp/ui_generation.log &
sleep 15
```

---

## 3. Pre-Flight Service Verification
Verify backend health and running process environment:
```bash
# 3.1 Backend health (Expect: backend: 200)
curl -s -o /dev/null -w "backend: %{http_code}\n" http://localhost:8100/health

# 3.2 Verify vLLM is responding (Expect: 200)
curl -s -o /dev/null -w "vLLM: %{http_code}\n" http://localhost:8000/v1/models

# 3.3 Verify aion_api.py has the correct model
API_PID=$(pgrep -f aion_api.py | head -1)
if [ -n "$API_PID" ]; then
    cat /proc/$API_PID/environ | tr '\0' '\n' | grep AION_MODEL
else
    echo "ERROR: aion_api.py is not running!"
fi
```

---

## 4. Run `quickstart.py` Diagnostic Test
Before touching the UI or API endpoints, verify the entire v3 pipeline in isolation:
```bash
conda activate aion 2>/dev/null || source activate aion 2>/dev/null
python quickstart.py 2>&1 | tee /tmp/quickstart.log
```
Check generated outputs:
```bash
cat quickstart_output.json | python -m json.tool | head -35
```
Expected output:
- Step 1: PASS (vLLM reachable)
- Step 2: PASS (Groq reachable)
- Step 3: PASS (Evidence indexed)
- Step 4: PASS (10 questions generated, 0 unresolved, paper_status: PASS)

---

## 5. Direct Endpoint Verification (`/api/v3/generate`)
Test the synchronous generation endpoint directly:
```bash
curl -X POST http://localhost:8100/api/v3/generate \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "IOT Agriculture and Health care",
    "exam_type": "IAT1",
    "file_ids": ["b56ada8d-c9a", "e59d96e4-d0b", "081fcfb8-bf2"]
  }' | python -m json.tool | head -40
```
Expected: `{"success": true, "paper_status": "PASS", "questions": [...], "modules": [...]}`.

---

## 6. Launch Frontend & Test UI
Start the web client (runs on port 5174 or port 3000):
```bash
./aiq frontend 5174 2>&1 | tee /tmp/frontend.log &
sleep 5
curl -s -o /dev/null -w "frontend: %{http_code}\n" http://localhost:5174/
```
In browser:
1. Open `http://localhost:5174` (or port forwarded `3000`).
2. Upload the 3 IoT PDFs (or select existing upload IDs).
3. Select `IAT1`.
4. Click Generate and monitor `/tmp/ui_generation.log`.

---

## 7. Emergency Rollback Procedures
If any regressions occur during server testing:
```bash
# View recent commits
git log -5 --oneline

# Rollback target: last verified PASS milestone commit
git reset --hard 0434fc0df

# Restart backend
./aiq stop && ./aiq backend &
```
Golden reference commit: `0434fc0df` (or `6cb75c4e` for quickstart + /api/v3/generate additions).
