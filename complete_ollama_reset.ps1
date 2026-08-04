# complete_ollama_reset.ps1
# Complete Ollama nuclear reinstall script for Windows

Write-Host "=== OLLAMA COMPLETE RESET ===" -ForegroundColor Yellow

# 1. Kill processes
Write-Host "[1/6] Killing processes..."
taskkill /F /IM ollama.exe 2>$null
taskkill /F /IM ollama_llama_server.exe 2>$null
Start-Sleep -Seconds 3

# 2. Uninstall
Write-Host "[2/6] Uninstalling Ollama..."
winget uninstall Ollama --silent

# 3. Delete data
Write-Host "[3/6] Deleting all Ollama data..."
$paths = @(
    "$env:USERPROFILE\.ollama",
    "$env:LOCALAPPDATA\Ollama",
    "$env:LOCALAPPDATA\Programs\Ollama"
)
foreach ($path in $paths) {
    if (Test-Path $path) {
        Remove-Item $path -Recurse -Force
        Write-Host "  Deleted: $path"
    }
}

# 4. Reinstall
Write-Host "[4/6] Reinstalling Ollama..."
winget install Ollama

# 5. Wait for installation
Start-Sleep -Seconds 10

# 6. Start service
Write-Host "[5/6] Starting Ollama..."
Start-Process "ollama" -ArgumentList "serve"
Start-Sleep -Seconds 5

# 7. Pull model
Write-Host "[6/6] Pulling qwen2.5:1.5b..."
ollama pull qwen2.5:1.5b

Write-Host ""
Write-Host "=== RESET COMPLETE ===" -ForegroundColor Green
Write-Host ""
Write-Host "Test with: python diagnose_ollama.py"
