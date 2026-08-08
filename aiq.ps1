<#
.SYNOPSIS
    AIQ — Academic Intelligence Questioning
    PowerShell launcher for Windows development

.USAGE
    .\aiq.ps1                  # start everything
    .\aiq.ps1 backend          # backend only
    .\aiq.ps1 frontend         # frontend only
    .\aiq.ps1 status           # check services
    .\aiq.ps1 stop             # stop all
    .\aiq.ps1 model            # show model config
    .\aiq.ps1 logs             # tail backend log
#>

param([string]$Command = "")

$ROOT         = $PSScriptRoot
$BACKEND_PORT = 8100
$FRONTEND_DIR = "$ROOT\frontend\artifacts\qp-generator"
$LOG_DIR      = "$ROOT\logs"
$BACKEND_LOG  = "$LOG_DIR\backend.log"

New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

# ── Helpers ───────────────────────────────────────────────────────────────────

function Write-AIQ  { param($msg) Write-Host "[AIQ] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "[AIQ] $msg" -ForegroundColor Yellow }
function Write-Err  { param($msg) Write-Host "[AIQ ERROR] $msg" -ForegroundColor Red }

function Show-Banner {
    Write-Host ""
    Write-Host "  +---------------------------------------+" -ForegroundColor Cyan
    Write-Host "  |          AIQ -- Academic IQ           |" -ForegroundColor Cyan
    Write-Host "  |   Academic Intelligence Questioning    |" -ForegroundColor Cyan
    Write-Host "  +---------------------------------------+" -ForegroundColor Cyan
    Write-Host ""
}

function Invoke-VEnv {
    $venvPaths = @(
        "$ROOT\.venv\Scripts\Activate.ps1",
        "$ROOT\venv\Scripts\Activate.ps1"
    )
    foreach ($p in $venvPaths) {
        if (Test-Path $p) {
            & $p
            Write-AIQ "Virtual environment activated"
            return
        }
    }
    Write-Warn "No venv found - using system Python"
}

function Get-ResolvedModel {
    try {
        python -c 'import core.config.production_model as p; print(p.get_production_model())'
    } catch {
        return "qwen2.5:3b (unavailable)"
    }
}

function Test-Port {
    param([int]$Port)
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $c.Connect("127.0.0.1", $Port)
        $c.Close()
        return $true
    } catch {
        return $false
    }
}

function Load-EnvFile {
    param([string]$Device)
    $envFile = "$ROOT\.env.$Device"
    if (Test-Path $envFile) {
        Get-Content $envFile |
            Where-Object { $_ -match '^\s*[^#]\S+=\S' } |
            ForEach-Object {
                $k, $v = $_ -split "=", 2
                [System.Environment]::SetEnvironmentVariable(
                    $k.Trim(), $v.Trim(), "Process"
                )
            }
        Write-AIQ "Loaded: .env.$Device"
    }
}

# ── Commands ──────────────────────────────────────────────────────────────────

function Start-Backend {
    Invoke-VEnv
    Set-Location $ROOT
    $model = Get-ResolvedModel
    Write-AIQ "Model: $model"
    Write-AIQ "Starting backend on port $BACKEND_PORT..."
    Start-Process -FilePath "python" -ArgumentList "aion_api.py" -RedirectStandardOutput $BACKEND_LOG -RedirectStandardError "$LOG_DIR\backend_err.log" -NoNewWindow

    $retries = 0
    while (-not (Test-Port $BACKEND_PORT)) {
        Start-Sleep -Seconds 1
        $retries++
        Write-Host "." -NoNewline
        if ($retries -ge 30) {
            Write-Host ""
            Write-Err "Backend did not start in 30s. Check: $BACKEND_LOG"
            return $false
        }
    }
    Write-Host ""
    Write-AIQ "Backend ready at http://localhost:$BACKEND_PORT"
    return $true
}

function Start-Frontend {
    if (-not (Test-Path $FRONTEND_DIR)) {
        Write-Err "Frontend not found: $FRONTEND_DIR"
        return
    }
    if (Test-Port -Port 5173) {
        Write-Warn "Port 5173 occupied - clearing existing process..."
        $conn = Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue
        if ($conn) { Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue }
        Start-Sleep -Seconds 1
    }
    Set-Location $FRONTEND_DIR
    Write-AIQ "Starting frontend..."
    Start-Process "cmd.exe" -ArgumentList "/c pnpm run dev --host" -NoNewWindow
    Start-Sleep -Seconds 4
    Write-AIQ "Frontend starting - check output for port"
}

function Invoke-Start {
    Show-Banner

    # Device detection
    $device = $env:AION_DEVICE
    if (-not $device) {
        $freeGB = [math]::Round(
            (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1
        )
        if     ($freeGB -ge 30) { $device = "server" }
        elseif ($freeGB -ge 10) { $device = "desktop" }
        else                    { $device = "laptop" }
        $ramStr = [string]$freeGB + " GB free RAM"
        Write-AIQ "Auto-detected: $device ($ramStr)"
        $env:AION_DEVICE = $device
    }

    Load-EnvFile -Device $device

    Set-Location $ROOT

    $ok = Start-Backend
    if ($ok) { Start-Frontend }

    Write-Host ""
    Write-Host "================================================" -ForegroundColor Green
    Write-Host "  AIQ is running" -ForegroundColor Green
    Write-Host "  Backend  : http://localhost:$BACKEND_PORT" -ForegroundColor Green
    Write-Host "  Device   : $device" -ForegroundColor Green
    Write-Host "  Log      : $BACKEND_LOG" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green
    Write-Host ""
    Write-AIQ "Press Ctrl+C to stop"
    try { while ($true) { Start-Sleep -Seconds 5 } }
    finally { Invoke-Stop }
}

function Invoke-Stop {
    Write-AIQ "Stopping AIQ..."

    # Kill backend by port
    $conn = Get-NetTCPConnection -LocalPort $BACKEND_PORT -ErrorAction SilentlyContinue
    if ($conn) {
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-AIQ "Backend stopped"
    }

    # Kill frontend
    foreach ($port in @(5173, 5174)) {
        $conn = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
        if ($conn) {
            Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
            Write-AIQ "Frontend stopped (port $port)"
        }
    }
}

function Invoke-Status {
    Write-Host ""
    Write-Host "AIQ Status" -ForegroundColor Cyan

    $services = @(
        @{ Name="Backend";  Port=$BACKEND_PORT },
        @{ Name="Ollama";   Port=11434 },
        @{ Name="Frontend"; Port=5173 }
    )

    foreach ($svc in $services) {
        $ok = Test-Port -Port $svc.Port
        $icon  = if ($ok) { "OK" } else { "DOWN" }
        $color = if ($ok) { "Green" } else { "Red" }
        $state = if ($ok) { "running" } else { "stopped" }
        $sName = $svc.Name
        $sPort = $svc.Port
        Write-Host "  $sName : $state [$icon] (port $sPort)" -ForegroundColor $color
    }

    # Health check
    if (Test-Port -Port $BACKEND_PORT) {
        try {
            $h = Invoke-RestMethod "http://localhost:$BACKEND_PORT/api/v1/health" -TimeoutSec 3
            $hMod = $h.resolved_model
            $hSrc = $h.model_source
            Write-Host "  Model     : $hMod (source: $hSrc)" -ForegroundColor Cyan
        } catch {}
    }
    Write-Host ""
}

function Invoke-Model {
    Invoke-VEnv
    Set-Location $ROOT
    Write-Host ""
    Write-Host "AIQ Model Configuration" -ForegroundColor Cyan
    python -c 'import core.config.production_model as p; print(p.get_production_model())'
}

function Invoke-Logs {
    if (Test-Path $BACKEND_LOG) {
        Get-Content $BACKEND_LOG -Wait
    } else {
        Write-Err "No log at $BACKEND_LOG"
        Write-Err "Start AIQ first: .\aiq.ps1"
    }
}

function Show-Help {
    Show-Banner
    Write-Host "Usage: .\aiq.ps1 [command]"
    Write-Host ""
    Write-Host "Commands:"
    Write-Host "  (none)     Start backend + frontend"
    Write-Host "  backend    Start backend only"
    Write-Host "  frontend   Start frontend only"
    Write-Host "  stop       Stop all AIQ processes"
    Write-Host "  status     Show running status"
    Write-Host "  logs       Tail backend logs"
    Write-Host "  model      Show model configuration"
    Write-Host "  help       Show this help"
    Write-Host ""
    Write-Host "Environment:"
    Write-Host "  AION_DEVICE=laptop|desktop|server"
    Write-Host "  AION_MODEL=qwen2.5:7b  (manual override)"
}

# ── Entry Point ───────────────────────────────────────────────────────────────

switch ($Command) {
    "backend"  { Invoke-VEnv; Set-Location $ROOT; Start-Backend; Read-Host "Press Enter to stop" }
    "frontend" { Start-Frontend; Read-Host "Press Enter to stop" }
    "stop"     { Invoke-Stop }
    "status"   { Invoke-Status }
    "model"    { Invoke-Model }
    "logs"     { Invoke-Logs }
    "help"     { Show-Help }
    ""         { Invoke-Start }
    default    { Write-Err "Unknown command: $Command"; Show-Help }
}
