# AION Startup — OpenVINO + Intel Arc (Laptop local testing)

Write-Host "`n=== AION OpenVINO Startup ===" -ForegroundColor Cyan

# Power plan
powercfg /setactive SCHEME_MIN 2>$null
Write-Host "[1/3] Power plan → High Performance"

# Check OpenVINO model
$ovModel = "C:\Users\Tarun J\OneDrive\Desktop\AION\models\qwen2.5-7b-ov"
$ovReady = (Test-Path "$ovModel\openvino_model.xml")

if ($ovReady) {
    Write-Host "[2/3] OpenVINO model found — using Intel Arc GPU" -ForegroundColor Green
    $env:AION_USE_OPENVINO = "1"
    $env:AION_OV_MODEL     = $ovModel
    $env:AION_MODEL        = "openvino"
} else {
    Write-Host "[2/3] OpenVINO model not found — using Ollama on CPU" -ForegroundColor Yellow
    Write-Host "      Run optimum-cli export to enable Arc GPU acceleration"

    # Start Ollama with minimal settings
    Get-Process -Name "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 3
    $env:OLLAMA_KEEP_ALIVE  = "2h"
    $env:OLLAMA_NUM_THREADS = "14"
    $env:AION_MODEL         = "qwen2.5:7b"
    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}

$free = [math]::Round(
    (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB, 1
)

Write-Host "[3/3] Starting AION..."
Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host " Backend  : http://localhost:8100"              -ForegroundColor Green
Write-Host " Model    : $(if ($ovReady) { 'Qwen2.5-7B (OpenVINO/Arc GPU)' } else { 'qwen2.5:7b (Ollama/CPU)' })" -ForegroundColor Green
Write-Host " Free RAM : ${free}GB"                          -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""

Set-Location "C:\Users\Tarun J\OneDrive\Desktop\AION"
python aion_api.py
