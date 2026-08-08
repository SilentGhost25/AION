@echo off
setlocal enabledelayedexpansion
title AIQ — Academic Intelligence Questioning

:: =============================================================
:: AIQ Windows Launcher
:: Usage:
::   aiq              → start everything
::   aiq backend      → backend only
::   aiq frontend     → frontend only
::   aiq status       → check services
::   aiq stop         → stop all
::   aiq model        → show model config
:: =============================================================

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "BACKEND_PORT=8100"
set "FRONTEND_DIR=%ROOT%\frontend\artifacts\qp-generator"
set "LOG_DIR=%ROOT%\logs"
set "PID_DIR=%ROOT%\.aiq"
set "BACKEND_LOG=%LOG_DIR%\backend.log"

if not exist "%LOG_DIR%"  mkdir "%LOG_DIR%"
if not exist "%PID_DIR%"  mkdir "%PID_DIR%"

set "CMD=%~1"
if "%CMD%"==""          goto :cmd_start
if "%CMD%"=="backend"   goto :cmd_backend
if "%CMD%"=="frontend"  goto :cmd_frontend
if "%CMD%"=="stop"      goto :cmd_stop
if "%CMD%"=="status"    goto :cmd_status
if "%CMD%"=="model"     goto :cmd_model
if "%CMD%"=="logs"      goto :cmd_logs
if "%CMD%"=="help"      goto :cmd_help
echo [AIQ ERROR] Unknown command: %CMD%
goto :cmd_help

:: ── Banner ────────────────────────────────────────────────────────────────────
:banner
echo.
echo   ╔═══════════════════════════════════════╗
echo   ║          AIQ — Academic IQ             ║
echo   ║   Academic Intelligence Questioning    ║
echo   ╚═══════════════════════════════════════╝
echo.
goto :eof

:: ── Activate venv ─────────────────────────────────────────────────────────────
:activate_venv
if exist "%ROOT%\.venv\Scripts\activate.bat" (
    call "%ROOT%\.venv\Scripts\activate.bat"
    echo [AIQ] Virtual environment activated
) else if exist "%ROOT%\venv\Scripts\activate.bat" (
    call "%ROOT%\venv\Scripts\activate.bat"
    echo [AIQ] Virtual environment activated (venv\)
) else (
    echo [AIQ] WARNING: No venv found — using system Python
)
goto :eof

:: ── Start Everything ──────────────────────────────────────────────────────────
:cmd_start
call :banner

:: Auto-detect device from env or default to laptop
if defined AION_DEVICE goto :load_env
set "AION_DEVICE=laptop"

:: Try to load env file
:load_env
set "ENVFILE=%ROOT%\.env.%AION_DEVICE%"
if exist "%ENVFILE%" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%ENVFILE%") do (
        if not "%%A"=="" if not "%%A:~0,1%"=="#" (
            set "%%A=%%B"
        )
    )
    echo [AIQ] Loaded: .env.%AION_DEVICE%
)

call :activate_venv
cd /d "%ROOT%"

:: Resolve model
echo [AIQ] Resolving model...
python -c "from core.config.production_model import get_production_model,get_resolution_info; m=get_production_model(); i=get_resolution_info(); print(f'[AIQ] Model: {m} (source: {i[chr(34)+\"source\"+chr(34)]})')" 2>nul

:: Start backend in new window
echo [AIQ] Starting backend...
start "AIQ Backend" cmd /k "cd /d "%ROOT%" && python aion_api.py"

:: Wait for backend
echo [AIQ] Waiting for backend...
:wait_backend
timeout /t 2 /nobreak >nul
powershell -Command "try { (New-Object Net.Sockets.TcpClient('127.0.0.1',%BACKEND_PORT%)).Close(); exit 0 } catch { exit 1 }" >nul 2>&1
if errorlevel 1 goto :wait_backend
echo [AIQ] Backend ready at http://localhost:%BACKEND_PORT%

:: Start frontend
echo [AIQ] Starting frontend...
cd /d "%FRONTEND_DIR%"
start "AIQ Frontend" cmd /k "pnpm run dev --host"

echo.
echo ================================================
echo   AIQ is running
echo   Backend  : http://localhost:%BACKEND_PORT%
echo   Device   : %AION_DEVICE%
echo ================================================
echo.
echo Both services started in separate windows.
echo Close those windows or run: aiq stop
goto :eof

:: ── Backend only ──────────────────────────────────────────────────────────────
:cmd_backend
call :activate_venv
cd /d "%ROOT%"
echo [AIQ] Starting backend on port %BACKEND_PORT%...
python aion_api.py
goto :eof

:: ── Frontend only ─────────────────────────────────────────────────────────────
:cmd_frontend
if not exist "%FRONTEND_DIR%" (
    echo [AIQ ERROR] Frontend not found: %FRONTEND_DIR%
    exit /b 1
)
cd /d "%FRONTEND_DIR%"
echo [AIQ] Starting frontend...
pnpm run dev --host
goto :eof

:: ── Stop ──────────────────────────────────────────────────────────────────────
:cmd_stop
echo [AIQ] Stopping AIQ services...

:: Kill backend by port
for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%BACKEND_PORT% "') do (
    if not "%%P"=="0" (
        taskkill /F /PID %%P >nul 2>&1
        echo [AIQ] Backend stopped (PID %%P)
    )
)

:: Kill Vite dev server (port 5173/5174)
for %%PORT in (5173 5174) do (
    for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%%PORT "') do (
        if not "%%P"=="0" (
            taskkill /F /PID %%P >nul 2>&1
            echo [AIQ] Frontend stopped (PID %%P)
        )
    )
)

echo [AIQ] Done
goto :eof

:: ── Status ────────────────────────────────────────────────────────────────────
:cmd_status
echo.
echo [AIQ] Status Check
echo.

:: Backend
powershell -Command "try { (New-Object Net.Sockets.TcpClient('127.0.0.1',%BACKEND_PORT%)).Close(); Write-Host '  Backend   ● running  (port %BACKEND_PORT%)' -ForegroundColor Green } catch { Write-Host '  Backend   ● stopped' -ForegroundColor Red }"

:: Ollama
powershell -Command "try { (New-Object Net.Sockets.TcpClient('127.0.0.1',11434)).Close(); Write-Host '  Ollama    ● running  (port 11434)' -ForegroundColor Green } catch { Write-Host '  Ollama    ● stopped' -ForegroundColor Red }"

:: Frontend
powershell -Command "try { (New-Object Net.Sockets.TcpClient('127.0.0.1',5173)).Close(); Write-Host '  Frontend  [OK]  (port 5173)' -ForegroundColor Green } catch { Write-Host '  Frontend  [DOWN]' -ForegroundColor Red }"

echo.
goto :eof

:: ── Model ─────────────────────────────────────────────────────────────────────
:cmd_model
call :activate_venv
cd /d "%ROOT%"
echo.
echo [AIQ] Model Configuration
echo.
python -c "import core.config.production_model as p; print('  Resolved :', p.get_production_model()); print('  Source   :', p.get_resolution_info().get('source')); print('  Device   :', p.get_resolution_info().get('device'))"
goto :eof

:: ── Logs ──────────────────────────────────────────────────────────────────────
:cmd_logs
if exist "%BACKEND_LOG%" (
    powershell -Command "Get-Content '%BACKEND_LOG%' -Wait"
) else (
    echo [AIQ ERROR] No log file at %BACKEND_LOG%
    echo Start AIQ first: aiq
)
goto :eof

:: ── Help ──────────────────────────────────────────────────────────────────────
:cmd_help
call :banner
echo Usage: aiq [command]
echo.
echo Commands:
echo   (none)     Start backend + frontend
echo   backend    Start backend only
echo   frontend   Start frontend only
echo   stop       Stop all AIQ processes
echo   status     Show running status
echo   logs       Tail backend logs
echo   model      Show model configuration
echo   help       Show this help
echo.
echo Environment:
echo   AION_DEVICE=laptop^|desktop^|server
echo   AION_MODEL=qwen2.5:7b
echo.
echo Examples:
echo   aiq
echo   set AION_DEVICE=server ^&^& aiq
echo   aiq status
echo   aiq stop
goto :eof

endlocal
