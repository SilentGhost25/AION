#!/bin/bash
# AION High-Throughput vLLM Server Launch Script (NVIDIA L40 48GB)
# Usage: bash start_vllm.sh [model_name] [port]

MODEL="${1:-${AION_MODEL:-Qwen/Qwen2.5-14B-Instruct}}"
PORT="${2:-8000}"
GPU_UTIL="${AION_GPU_UTILIZATION:-0.85}"
MAX_LEN=8192
CONCURRENCY="${AION_CONCURRENCY:-16}"

echo "================================================================"
echo "Starting vLLM High-Throughput Engine for AION"
echo "  Model       : $MODEL"
echo "  Port        : $PORT"
echo "  GPU Util    : $GPU_UTIL (leaves ~7.2 GB for dynamic concept embeddings)"
echo "  Max Context : $MAX_LEN"
echo "  Concurrency : $CONCURRENCY streams"
echo "================================================================"

# Configuration A (Production Default): Native n-gram prompt lookup speculative decoding
# - 0 MB extra VRAM, zero risk of missing draft head startup crashes
# - Yields 1.25x to 1.4x decode speedup on structured JSON tokens
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --enable-prefix-caching \
    --guided-decoding-backend xgrammar \
    --speculative-model [ngram] \
    --num-speculative-tokens 3 \
    --ngram-prompt-lookup-max 3 \
    --max-num-seqs "$CONCURRENCY" \
    --trust-remote-code

# Note: Configuration B (EAGLE3 draft head) can be activated by replacing --speculative-model [ngram]
# with: --speculative-model yuhuili/EAGLE-Qwen2.5-14B-Instruct
