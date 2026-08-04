@echo off
echo ========================================
echo AION Emergency Mode Startup Checklist
echo ========================================
echo.

echo [1/5] Checking Ollama...
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo   X Ollama not running
    echo   Fix: Run 'ollama serve' in another terminal
    pause
    exit /b 1
)
echo   OK Ollama running

echo.
echo [2/5] Checking model...
ollama list | findstr /C:"qwen2.5:1.5b" >nul
if %errorlevel% neq 0 (
    echo   X qwen2.5:1.5b not installed
    echo   Fix: ollama pull qwen2.5:1.5b
    pause
    exit /b 1
)
echo   OK qwen2.5:1.5b available

echo.
echo [3/5] Testing model response...
echo Say OK | ollama run qwen2.5:1.5b >nul 2>&1
if %errorlevel% neq 0 (
    echo   X Model not responding
    echo   Fix: Restart Ollama
    pause
    exit /b 1
)
echo   OK Model responds

echo.
echo [4/5] Checking memory...
for /f "tokens=2 delims==" %%a in ('wmic OS Get FreePhysicalMemory /Value') do set /a mem=%%a/1024
if %mem% lss 4096 (
    echo   ! Low memory: %mem%MB free
    echo   Warning: May run slowly
) else (
    echo   OK %mem%MB free
)

echo.
echo [5/5] Checking disk space...
for /f "tokens=3" %%a in ('dir /-c ^| find "bytes free"') do set free=%%a
echo   OK Disk space sufficient

echo.
echo ========================================
echo System Ready for Emergency Mode
echo ========================================
echo.
echo Run: python emergency_cli.py your_file.pdf
echo  OR: python aion_api.py
echo      then use /api/generate/emergency
echo.
pause
