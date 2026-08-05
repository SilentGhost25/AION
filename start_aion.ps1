# ═══════════════════════════════════════════════════════════════
# AION Startup Script — Optimized for Intel Core Ultra 5 125H
#                        + Intel Arc + 16GB RAM
# ═══════════════════════════════════════════════════════════════

Write-Host "============================================"
Write-Host " AION Startup — Optimizing for your hardware"
Write-Host "============================================`n"

# ── Step 1: Set High Performance power plan ───────────────────
Write-Host "[1/6] Setting High Performance power plan..."
$hp = powercfg /list | Select-String "High performance"
if ($hp) {
    $guid = ($hp -split "\s+")[3]
    powercfg /setactive $guid 2>$null
    Write-Host "      Power plan: High Performance"
}

# ── Step 2: Free RAM ──────────────────────────────────────────
Write-Host "[2/6] Freeing RAM..."
$before = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)

$killList = @("chrome","msedge","firefox","slack","teams","discord","SearchIndexer")
foreach ($name in $killList) {
    Stop-Process -Name $name -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2

$after = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)
Write-Host "      RAM: ${before}GB → ${after}GB free"

# ── Step 3: Set Ollama environment variables ──────────────────
Write-Host "[3/6] Configuring Ollama..."

# Use all 18 logical threads
$env:OLLAMA_NUM_THREADS       = "18"

# Keep model in RAM for 60 minutes
$env:OLLAMA_KEEP_ALIVE        = "60m"

# Flash attention (faster on Arc GPU)
$env:OLLAMA_FLASH_ATTENTION   = "1"

# Intel Arc GPU offloading
$env:OLLAMA_GPU_LAYERS        = "20"

# Queue settings
$env:OLLAMA_MAX_QUEUE         = "5"
$env:OLLAMA_NUM_PARALLEL      = "1"

# Choose model based on available RAM
$freeRAM = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,1)
Write-Host "      Free RAM: ${freeRAM}GB"

if ($freeRAM -ge 5.5) {
    $env:AION_MODEL = "aion-exam"
    Write-Host "      Model: aion-exam (7B fine-tuned)"
} elseif ($freeRAM -ge 3.0) {
    $env:AION_MODEL = "qwen2.5:3b"
    Write-Host "      Model: qwen2.5:3b (RAM constraint)"
} else {
    $env:AION_MODEL = "qwen2.5:3b"
    Write-Host "      WARNING: Very low RAM — using qwen2.5:3b"
    Write-Host "      Close more apps for better performance"
}

# ── Step 4: Restart Ollama with new settings ──────────────────
Write-Host "[4/6] Restarting Ollama with optimized settings..."
Stop-Process -Name "ollama" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
Start-Sleep -Seconds 5
Write-Host "      Ollama started"

# ── Step 5: Pre-load model ────────────────────────────────────
Write-Host "[5/6] Pre-loading $env:AION_MODEL into RAM..."
Write-Host "      This takes 30-90 seconds on first load..."

$body = @{
    model   = $env:AION_MODEL
    prompt  = "ready"
    stream  = $false
    options = @{ num_predict = 1 }
} | ConvertTo-Json -Depth 5

$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
    Invoke-RestMethod `
        -Uri    "http://127.0.0.1:11434/api/generate" `
        -Method POST `
        -ContentType "application/json" `
        -Body   $body `
        -TimeoutSec 300 | Out-Null
    $sw.Stop()
    Write-Host "      Loaded in $([math]::Round($sw.Elapsed.TotalSeconds,1))s"
} catch {
    Write-Host "      WARNING: Pre-load failed — first request will be slow"
}

# ── Step 6: Start AION ────────────────────────────────────────
Write-Host "[6/6] Starting AION backend..."
Write-Host ""
Write-Host "============================================"
Write-Host " AION ready at http://localhost:8100"
Write-Host " Model: $env:AION_MODEL"
Write-Host " Frontend: http://localhost:5174"
Write-Host "============================================`n"

Set-Location "C:\Users\Tarun J\OneDrive\Desktop\AION"
python aion_api.py
