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
    Write-Host "  ╔═══════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "  ║          AIQ — Academic IQ             ║" -ForegroundColor Cyan
