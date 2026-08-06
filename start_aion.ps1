# AION v1.0 Stable Startup Script

Write-Host "`n=== AION v1.0 Stable ===" -ForegroundColor Cyan

# Free RAM
Write-Host "[1/4] Freeing RAM..."
@("brave","chrome","msedge","ChatGPT","language_server_windows_x64") |
    ForEach-Object { Stop-Process -Name $_ -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2
$free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)
Write-Host "      Free RAM: ${free} GB"

# Configure Ollama
Write-Host "[2/4] Starting Ollama..."
$env:OLLAMA_KEEP_ALIVE   = "2h"
$env:OLLAMA_NUM_THREADS  = "14"
$env:OLLAMA_NUM_PARALLEL = "1"
Stop-Process -Name "ollama" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
Start-Sleep -Seconds 5

# Pre-load model
Write-Host "[3/4] Loading qwen2.5:3b into RAM..."
$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
    Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/generate" -Method POST `
        -ContentType "application/json" `
        -Body (@{ model="qwen2.5:3b"; prompt="hi"; stream=$false; options=@{num_predict=1} } | ConvertTo-Json -Depth 5) `
        -TimeoutSec 120 | Out-Null
    $sw.Stop()
    Write-Host "      Loaded in $([math]::Round($sw.Elapsed.TotalSeconds,1))s"
} catch {
    Write-Host "      Warning: pre-load failed. First request will be slow."
}

# Start AION
Write-Host "[4/4] Starting AION backend..."
Write-Host ""
Write-Host "  Backend  : http://localhost:8100" -ForegroundColor Green
Write-Host "  Frontend : http://localhost:5174" -ForegroundColor Green
Write-Host "  Model    : qwen2.5:3b (stable)" -ForegroundColor Green
Write-Host ""

Set-Location "C:\Users\Tarun J\OneDrive\Desktop\AION"
$env:AION_MODEL = "qwen2.5:3b"
python aion_api.py
