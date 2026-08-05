# Run this once before starting aion_api.py
$body = @{
    model  = "aion-exam"
    prompt = "ready"
    stream = $false
} | ConvertTo-Json

Write-Host "Pre-loading aion-exam into RAM (this takes ~75 seconds)..."

Invoke-RestMethod `
    -Uri "http://127.0.0.1:11434/api/generate" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body `
    -TimeoutSec 300 | Out-Null

Write-Host "Model loaded. Starting AION..."
python "C:\Users\Tarun J\OneDrive\Desktop\AION\aion_api.py"
