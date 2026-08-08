# AION Startup — Environment-Aware
# Usage: .\start_aion.ps1 [laptop|desktop|server]

param([string]$device = "")

# Determine which env file to load
if ($device -eq "") {
    $freeGB = [math]::Round(
        (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1
    )
    if ($freeGB -ge 30)      { $device = "server" }
    elseif ($freeGB -ge 10)  { $device = "desktop" }
    else                      { $device = "laptop" }
    Write-Host "Auto-detected device: $device (${freeGB}GB free RAM)"
}

# Load environment file
$envFile = "$PSScriptRoot\.env.$device"
if (Test-Path $envFile) {
    Get-Content $envFile |
        Where-Object { $_ -match "^\s*[^#]\S+=\S" } |
        ForEach-Object {
            $k, $v = $_ -split "=", 2
            [System.Environment]::SetEnvironmentVariable(
                $k.Trim(), $v.Trim(), "Process"
            )
        }
    Write-Host "Loaded: .env.$device"
} else {
    $env:AION_DEVICE = $device
    Write-Host "No .env.$device found — using AION_DEVICE=$device"
}

# Show resolved model (from Python authority)
$resolvedModel = python -c "
from core.config.production_model import get_production_model, get_resolution_info
model = get_production_model()
info  = get_resolution_info()
print(f\"{model} (source: {info['source']})\")
" 2>$null

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host " AION Startup" -ForegroundColor Green
Write-Host " Device  : $device" -ForegroundColor Green
Write-Host " Model   : $resolvedModel" -ForegroundColor Green
Write-Host " Backend : http://localhost:8100" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""

# Start Ollama
$env:OLLAMA_KEEP_ALIVE = if ($env:OLLAMA_KEEP_ALIVE) { $env:OLLAMA_KEEP_ALIVE } else { "2h" }
Get-Process -Name "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3
Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
Start-Sleep -Seconds 6

Set-Location $PSScriptRoot
python aion_api.py
