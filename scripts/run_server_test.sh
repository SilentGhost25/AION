#!/usr/bin/env bash
# scripts/run_server_test.sh
# End-to-end execution of preflight gate, backend bootstrap, and direct v3 test.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== [1/4] Pinning Engine Environment ==="
export AION_BACKEND="vllm"
export AION_LLM_HOST="http://127.0.0.1:8000"

# Auto-resolve active served model ID directly from engine
if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    ACTIVE_MODEL="$(curl -sf http://127.0.0.1:8000/v1/models | jq -r '.data[0].id // empty')"
    if [ -n "$ACTIVE_MODEL" ]; then
        export AION_MODEL="$ACTIVE_MODEL"
        echo "Resolved active vLLM model: $AION_MODEL"
    fi
fi

echo "=== [2/4] Running Preflight Gate ==="
python scripts/preflight.py || {
    echo "❌ Preflight gate failed. Halting before pipeline execution."
    exit 1
}

echo "=== [3/4] Ensuring Clean Backend Process ==="
pkill -f "python.*(aion_api|llm_server)" 2>/dev/null || true
sleep 1

# Start backend in background
if [ -f "./aiq" ]; then
    ./aiq backend > /tmp/ui_generation.log 2>&1 &
else
    python aion_api.py > /tmp/ui_generation.log 2>&1 &
fi

echo "Waiting for backend on :8100..."
READY=0
for i in {1..30}; do
    if curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1 || curl -sf http://127.0.0.1:8100/ready >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 1
done

if [ "$READY" -ne 1 ]; then
    echo "❌ Backend failed to bind to port 8100 within 30s. Log output:"
    tail -n 40 /tmp/ui_generation.log
    exit 1
fi
echo "Backend is live on port 8100."

echo "=== [4/4] Executing Direct /api/v3/generate Request ==="
# Determine sample upload file IDs
SAMPLE_IDS="$(ls -1 workspace/uploads 2>/dev/null | head -3 | jq -R . | jq -s .)"
if [ -z "$SAMPLE_IDS" ] || [ "$SAMPLE_IDS" = "[]" ]; then
    SAMPLE_IDS='["b56ada8d-c9a","e59d96e4-d0b","081fcfb8-bf2"]'
fi

PAYLOAD=$(cat <<EOF
{
  "subject": "IOT Agriculture and Health care",
  "exam_type": "IAT1",
  "file_ids": $SAMPLE_IDS
}
EOF
)

echo "Payload:"
echo "$PAYLOAD" | jq .

curl -sS -X POST http://127.0.0.1:8100/api/v3/generate \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" | tee /tmp/last_run.json | jq '{success, paper_status, questions: (.questions | length), degraded: .degraded, blocked_reasons: .blocked_reasons}'
