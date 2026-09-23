# AION High-Throughput vLLM Server Launch Script (PowerShell / Windows / WSL)
# Usage: .\start_vllm.ps1 [-Model <name>] [-Port <port>]

param(
    [string]$Model = $(if ($env:AION_MODEL) { $env:AION_MODEL } else { "Qwen/Qwen2.5-14B-Instruct" }),
    [int]$Port = 8000,
    [double]$GpuUtil = $(if ($env:AION_GPU_UTILIZATION) { [double]$env:AION_GPU_UTILIZATION } else { 0.85 }),
    [int]$Concurrency = $(if ($env:AION_CONCURRENCY) { [int]$env:AION_CONCURRENCY } else { 16 })
)

Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "Starting vLLM High-Throughput Engine for AION" -ForegroundColor Cyan
Write-Host "  Model       : $Model" -ForegroundColor Gray
Write-Host "  Port        : $Port" -ForegroundColor Gray
Write-Host "  GPU Util    : $GpuUtil" -ForegroundColor Gray
Write-Host "  Concurrency : $Concurrency streams" -ForegroundColor Gray
Write-Host "================================================================" -ForegroundColor Cyan

python -m vllm.entrypoints.openai.api_server `
    --model $Model `
    --port $Port `
    --gpu-memory-utilization $GpuUtil `
    --max-model-len 8192 `
    --enable-prefix-caching `
    --guided-decoding-backend xgrammar `
    --speculative-model [ngram] `
    --num-speculative-tokens 3 `
    --ngram-prompt-lookup-max 3 `
    --max-num-seqs $Concurrency `
    --trust-remote-code
