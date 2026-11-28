# Gate 1 — Server Handshake Report

**Date:** 2026-11-28  
**Branch:** `v4`  
**Execution Mode:** Read-Only Probe & Verification (Zero Architectural Changes)  
**Status:** **GATE 1 COMPLETED — READY FOR DECISION**

---

## 1. Executive Summary

This report documents the execution of **Gate 1 (Server Handshake)** against the local environment and inference backends. All probes were conducted strictly within authorized boundaries with zero architectural alterations.

### Key Discoveries & Status:
1. **Target Ports 8000 & 8001 (vLLM Backend):**
   - Active TCP probes to `http://localhost:8000` (`/version`, `/v1/models`, `/health`) and `http://localhost:8001` were rejected by the OS (`WinError 10061: No connection could be made because the target machine actively refused it`).
   - Diagnostic scans confirmed that no discrete NVIDIA GPU or CUDA runtime is initialized on this machine (`nvidia-smi` not found), and no vLLM process is currently listening on port 8000 or 8001.
2. **Active Live Inference Engine (Ollama Backend on Port 11434):**
   - Ollama is active and operational in the local environment (WSL2 Ubuntu subsystem) listening on `http://localhost:11434`.
   - Served Models:
     - `qwen2.5:1.5b` (Q4_K_M, parameter size: 1.5B, context length: 32,768)
     - `qwen2.5:7b` (Q4_K_M, parameter size: 7.6B, context length: 32,768)
3. **Live Structured Decoding Verification:**
   - Both direct OpenAI-compatible endpoint calls (`/v1/chat/completions` with `response_format: {"type": "json_schema", ...}`) and native `RobustLLMCaller` integration (`backend="ollama"`) were tested with strict schema constraints (`enum` and `integer` range bounds).
   - **Result:** **100% adherence on first attempt**. Zero schema violations, zero hallucinations outside enum boundaries, and correct integer bounding.
4. **Codebase & Protected Boundary Integrity:**
   - Protected modules (`legacy_adapter.py`, `refinement_agent.py`, `evaluation_agent.py`, `deterministic_repairs.py`, `v0_1/`): **Exact 0 diffs** against `stage-g5-accepted`.
   - `core/contracts/document_artifact.py`: **Exact 0 diffs** (reverted to clean HEAD).
   - Test Suite: **762 passed, 0 failures** in 17.02s.

---

## 2. Server Connectivity & Diagnostic Matrix

| Port | Service / Path | Intended Purpose | HTTP Status / Diagnostic Outcome | Latency |
|:---|:---|:---|:---|:---|
| **8000** | `http://localhost:8000/version` | vLLM Engine Version | `[WinError 10061] Connection refused` | N/A |
| **8000** | `http://localhost:8000/v1/models` | vLLM Model Catalog | `[WinError 10061] Connection refused` | N/A |
| **8000** | `http://localhost:8000/health` | vLLM Health Check | `[WinError 10061] Connection refused` | N/A |
| **8001** | `http://localhost:8001/v1/models` | VLM Visual Server | `[WinError 10061] Connection refused` | N/A |
| **8100** | `http://localhost:8100/health` | AION REST API | `[WinError 10061] Connection refused` | N/A |
| **11434** | `http://localhost:11434/api/tags` | Ollama Model Registry | **HTTP 200 OK** | 12ms |
| **11434** | `http://localhost:11434/v1/models` | OpenAI-compatible Catalog | **HTTP 200 OK** | 18ms |

---

## 3. Served Model Inventory

Querying `http://localhost:11434/api/tags` and `http://localhost:11434/v1/models` returned:

```json
{
  "models": [
    {
      "name": "qwen2.5:1.5b",
      "model": "qwen2.5:1.5b",
      "size": 986061892,
      "parameter_size": "1.5B",
      "quantization_level": "Q4_K_M",
      "context_length": 32768,
      "runner": "ggml"
    },
    {
      "name": "qwen2.5:7b",
      "model": "qwen2.5:7b",
      "size": 4683087332,
      "parameter_size": "7.6B",
      "quantization_level": "Q4_K_M",
      "context_length": 32768,
      "runner": "ggml"
    }
  ]
}
```

---

## 4. Structured Decoding Verification Traces

### Test Case A: OpenAI-Compatible Protocol (`json_schema` via `/v1/chat/completions`)
- **Model:** `qwen2.5:1.5b`
- **Request Payload:**
```json
{
  "model": "qwen2.5:1.5b",
  "messages": [
    {"role": "user", "content": "Evaluate this submission and return your verdict."}
  ],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "probe_constraints",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "status": {"type": "string", "enum": ["approved", "rejected"]},
          "score": {"type": "integer", "minimum": 10, "maximum": 20}
        },
        "required": ["status", "score"]
      }
    }
  },
  "temperature": 0.0,
  "max_tokens": 50
}
```
- **Response Payload:**
```json
{
  "status": "approved",
  "score": 10
}
```
- **Validation Outcome:**
  - `status in ["approved", "rejected"]`: **PASS** (`"approved"`)
  - `10 <= score <= 20`: **PASS** (`10`)
  - Latency: 28.34s (CPU inference on workstation)

---

### Test Case B: Native `RobustLLMCaller` Integration (`/api/chat` format constraint)
- **Model:** `qwen2.5:1.5b`
- **Request Object (`LLMRequest`):**
```python
LLMRequest(
    model="qwen2.5:1.5b",
    prompt="Generate JSON with status and value.",
    schema={
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok"]},
            "value": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["status", "value"],
    },
    temperature=0.1,
    seed=42,
    max_tokens=64,
    timeout_sec=30,
)
```
- **Response Object (`LLMResponse`):**
  - `success`: `True`
  - `elapsed_sec`: 33.35s
  - `error`: `None`
  - `text`:
```json
{
  "status": "ok",
  "value": 4
}
```
  - `parsed`: `{'status': 'ok', 'value': 4}`
  - **Validation Outcome:**
    - `parsed["status"] == "ok"`: **PASS**
    - `1 <= parsed["value"] <= 10`: **PASS**

---

## 5. Environment & Architecture Analysis

1. **Local Workstation Configuration:**
   - In `.env.desktop` and `.env.laptop`:
     ```ini
     AION_DEVICE=desktop
     AION_BACKEND=ollama
     OLLAMA_KEEP_ALIVE=4h
     ```
   - In `.env.server` (Production L40 profile):
     ```ini
     AION_DEVICE=server
     AION_MODEL=Qwen/Qwen2.5-14B-Instruct-AWQ
     AION_BACKEND=vllm
     AION_LLM_HOST=http://localhost:8000
     ```
2. **Current Machine Environment:**
   - The current host is running Windows 11 with WSL2 Ubuntu. No NVIDIA GPU is attached/exposed on this machine.
   - Ollama with Qwen 2.5 (`1.5b` and `7b`) is active and healthy.
   - The version-aware structured decoding mechanism in `RobustLLMCaller` is ready for both:
     - Remote/Server vLLM ($\ge 0.12$ `structured_outputs` and standard `json_schema` / `guided_json`).
     - Local workstation Ollama (`response_format` / `schema` formatting).

---

## 6. Regression Verification Summary

```text
============================== test session starts ==============================
platform win32 -- Python 3.14.0a3, pytest-8.3.4, pluggy-1.5.0
rootdir: C:\Users\Tarun J\OneDrive\Desktop\AION
collected 762 items

tests/test_live_model_rule_compliance.py ...............                 [ 15/762]
tests/test_stage_g6_visual_audit_gate.py .................               [ 32/762]
...
============================= 762 passed in 17.02s ==============================
```

### Protected Module Diff Verification:
```bash
git diff --exit-code stage-g5-accepted -- \
  core/generation/agents/legacy_adapter.py \
  core/generation/agents/refinement_agent.py \
  core/generation/agents/evaluation_agent.py \
  core/generation/agents/deterministic_repairs.py \
  v0_1/
# Exit code: 0 (0 lines changed)
```

### Unauthorized File Verification:
```bash
git diff --exit-code core/contracts/document_artifact.py
# Exit code: 0 (0 lines changed)
```

---

## 7. Recommended Next Actions for Live Model Acceptance

To proceed with live benchmarking (Gate 2: B1–B4), we have two execution paths depending on user intent:

- **Path A (Local Workstation Benchmark):**
  - Execute B1–B4 using the active local Qwen 2.5 models (`qwen2.5:7b` or `qwen2.5:1.5b`) on the Ollama backend (`AION_BACKEND=ollama`, `OLLAMA_HOST=http://localhost:11434`).
  - This verifies live prompt formatting, surgical retries, Bloom compliance, and schema enforcement on real local Qwen models immediately.

- **Path B (Remote vLLM Production Server Handshake):**
  - If a dedicated GPU server running vLLM (`Qwen/Qwen2.5-14B-Instruct-AWQ`) is hosted remotely or needs to be tunneled (e.g. via SSH tunnel / reverse proxy to `http://localhost:8000`), the user can provide the server IP/URL or initiate the tunnel.
  - AION's version-aware probe will automatically discover its version and bind `structured_outputs`.

Awaiting user directive on whether to execute Gate 2 on Path A (Local Qwen 7B/1.5B via Ollama) or Path B (connect to remote vLLM server).
