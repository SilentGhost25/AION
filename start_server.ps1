# AION Server Start — loads server config only
# Usage: .\start_server.ps1
# Does NOT affect laptop dev environment

param([switch]$BackendOnly)

Write-Host "=== AION Server Mode ===" -ForegroundColor Cyan

# Load server env — only for this session
$envFile = "$PSScriptRoot\.env.server"
if (Test-Path $envFile) {
    Get-Content $envFile |
        Where-Object { $_ -match "^\s*[^#]\S+=\S" } |
        ForEach-Object {
            $k, $v = $_ -split "=", 2
            [System.Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), "Process")
            Write-Host "  $($k.Trim()) = $($v.Trim())" -ForegroundColor Gray
        }
    Write-Host ""
}

# Show resolved model
$model = python -c "
from core.config.production_model import get_production_model, get_resolution_info
m = get_production_model()
i = get_resolution_info()
print(f'{m} ({i[chr(115)+chr(111)+chr(117)+chr(114)+chr(99)+chr(101)]})')
" 2>$null

Write-Host "Model   : $model" -ForegroundColor Green
Write-Host "Device  : server (L40)" -ForegroundColor Green
Write-Host "Backend : http://localhost:8100" -ForegroundColor Green
Write-Host ""

Set-Location $PSScriptRoot
python aion_api.py
