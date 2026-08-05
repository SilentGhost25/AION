# 1. Kill everything eating RAM
Write-Host "🚀 Entering Turbo Mode..."
$hogs = @("brave", "chrome", "msedge", "Code", "ChatGPT", "slack", "teams")
foreach ($app in $hogs) { Stop-Process -Name $app -Force -ErrorAction SilentlyContinue }

# 2. Force Windows to clear standby memory
[System.GC]::Collect()

# 3. Set High Performance Power
powercfg /setactive SCHEME_MIN

# 4. Start Ollama with IGPU Priority (Intel Arc)
$env:OLLAMA_INTEL_GPU = "1"
$env:OLLAMA_NUM_THREADS = "14"
$env:OLLAMA_KEEP_ALIVE = "1h" 
Stop-Process -Name "ollama" -Force -ErrorAction SilentlyContinue
Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
Start-Sleep -Seconds 5

# 5. Pre-warm the model
Write-Host "🧠 Warming up AION-EXAM..."
ollama run aion-exam "ready" | Out-Null

# 6. Start Backend
Write-Host "✅ System Optimized. Starting AION..."
python aion_api.py
