@echo off
setlocal enabledelayedexpansion
title AIQ — Academic Intelligence Questioning

:: =============================================================
:: AIQ Windows Launcher
:: Usage:
::   aiq              → start everything (dynamic port allocation)
::   aiq [port]       → start everything with preferred backend port
::   aiq backend [p]  → backend only
::   aiq frontend [p] → frontend only
::   aiq status       → check services (verifies process identity)
::   aiq stop         → stop all AIQ processes safely
::   aiq model        → show model config
::   aiq logs         → tail backend logs
::   aiq help         → show help
:: =============================================================

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "DEFAULT_BACKEND_PORT=8100"
set "DEFAULT_FRONTEND_PORT=5174"
set "FRONTEND_DIR=%ROOT%\frontend\artifacts\qp-generator"
set "LOG_DIR=%ROOT%\logs"
set "PID_DIR=%ROOT%\.aiq"
set "BACKEND_LOG=%LOG_DIR%\backend.log"

if not exist "%LOG_DIR%"  mkdir "%LOG_DIR%"
if not exist "%PID_DIR%"  mkdir "%PID_DIR%"

set "CMD=%~1"
set "ARG2=%~2"

:: Check if first argument is a numeric port (e.g. `aiq 8105`)
echo %CMD%| findstr /r "^[0-9][0-9]*$" >nul
if not errorlevel 1 (
    set "PORT_OVERRIDE=%CMD%"
    set "CMD="
    goto :cmd_start
)

:: Check if second argument is a numeric port (e.g. `aiq backend 8105`)
if not "%ARG2%"=="" (
    echo %ARG2%| findstr /r "^[0-9][0-9]*$" >nul
    if not errorlevel 1 (
        set "PORT_OVERRIDE=%ARG2%"
    )
)

if "%CMD%"==""          goto :cmd_start
if "%CMD%"=="start"     goto :cmd_start
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
echo   +---------------------------------------+
echo   ^|          AIQ -- Academic IQ           ^|
echo   ^|   Academic Intelligence Questioning   ^|
echo   +---------------------------------------+
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
    echo [AIQ] WARNING: No venv found - using system Python
)
goto :eof

:: ── Read Configured Ports ────────────────────────────────────────────────────
:read_ports
if exist "%PID_DIR%\ports.env" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%PID_DIR%\ports.env") do (
        if not "%%A"=="" set "%%A=%%B"
    )
)
if not defined BACKEND_PORT if defined AION_PORT set "BACKEND_PORT=%AION_PORT%"
if not defined BACKEND_PORT set "BACKEND_PORT=%DEFAULT_BACKEND_PORT%"
if not defined FRONTEND_PORT set "FRONTEND_PORT=%DEFAULT_FRONTEND_PORT%"
goto :eof

:: ── Allocate Dynamic Ports ──────────────────────────────────────────────────
:allocate_ports
:: Base backend port priority: CLI argument > AION_PORT > BACKEND_PORT > 8100
if defined PORT_OVERRIDE (
    set "DESIRED_BACKEND_PORT=%PORT_OVERRIDE%"
) else if defined AION_PORT (
    set "DESIRED_BACKEND_PORT=%AION_PORT%"
) else if defined BACKEND_PORT (
    set "DESIRED_BACKEND_PORT=%BACKEND_PORT%"
) else (
    set "DESIRED_BACKEND_PORT=%DEFAULT_BACKEND_PORT%"
)

:: Base frontend port priority: FRONTEND_PORT_OVERRIDE > FRONTEND_PORT > 5174
if defined FRONTEND_PORT_OVERRIDE (
    set "DESIRED_FRONTEND_PORT=%FRONTEND_PORT_OVERRIDE%"
) else if defined FRONTEND_PORT (
    set "DESIRED_FRONTEND_PORT=%FRONTEND_PORT%"
) else (
    set "DESIRED_FRONTEND_PORT=%DEFAULT_FRONTEND_PORT%"
)

:: Dynamic backend port check via python
for /f "delims=" %%P in ('python -c "from core.config.server_config import find_available_port; print(find_available_port(!DESIRED_BACKEND_PORT!))" 2^>nul') do (
    set "RESOLVED_BACKEND=%%P"
)
if not defined RESOLVED_BACKEND (
    set "RESOLVED_BACKEND=!DESIRED_BACKEND_PORT!"
)
if not "!RESOLVED_BACKEND!"=="!DESIRED_BACKEND_PORT!" (
    echo [AIQ] Notice: Port !DESIRED_BACKEND_PORT! is occupied. Dynamically selected port !RESOLVED_BACKEND!
)
set "BACKEND_PORT=!RESOLVED_BACKEND!"
set "AION_PORT=!RESOLVED_BACKEND!"

:: Dynamic frontend port check via python
for /f "delims=" %%P in ('python -c "from core.config.server_config import find_available_port; print(find_available_port(!DESIRED_FRONTEND_PORT!))" 2^>nul') do (
    set "RESOLVED_FRONTEND=%%P"
)
if not defined RESOLVED_FRONTEND (
    set "RESOLVED_FRONTEND=!DESIRED_FRONTEND_PORT!"
)
if not "!RESOLVED_FRONTEND!"=="!DESIRED_FRONTEND_PORT!" (
    echo [AIQ] Notice: Port !DESIRED_FRONTEND_PORT! is occupied. Dynamically selected port !RESOLVED_FRONTEND!
)
set "FRONTEND_PORT=!RESOLVED_FRONTEND!"

:: Write active ports to .aiq\ports.env for status/stop synchronization
(
    echo BACKEND_PORT=!BACKEND_PORT!
    echo AION_PORT=!AION_PORT!
    echo FRONTEND_PORT=!FRONTEND_PORT!
) > "%PID_DIR%\ports.env"
goto :eof

:: ── Start Everything ──────────────────────────────────────────────────────────
:cmd_start
call :banner

:: Auto-detect device from env or default to laptop
if defined AION_DEVICE goto :load_env
set "AION_DEVICE=laptop"

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

:: Allocate dynamic ports (verifying availability and avoiding collisions)
call :allocate_ports

:: Resolve model
echo [AIQ] Resolving model...
python -c "from core.config.production_model import get_production_model,get_resolution_info; m=get_production_model(); i=get_resolution_info(); print(f'[AIQ] Model: {m} (source: {i[chr(34)+\"source\"+chr(34)]})')" 2>nul

:: Start backend in new window with explicit port env vars
echo [AIQ] Starting backend on port %BACKEND_PORT%...
start "AIQ Backend" cmd /k "cd /d "%ROOT%" && set "AION_PORT=%BACKEND_PORT%" && set "BACKEND_PORT=%BACKEND_PORT%" && python aion_api.py"

:: Wait for backend to be ready via health check
echo [AIQ] Waiting for backend...
:wait_backend
timeout /t 2 /nobreak >nul
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:%BACKEND_PORT%/api/health' -TimeoutSec 2 -UseBasicParsing; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { try { (New-Object Net.Sockets.TcpClient('127.0.0.1',%BACKEND_PORT%)).Close(); exit 0 } catch { exit 1 } }" >nul 2>&1
if errorlevel 1 goto :wait_backend
echo [AIQ] Backend ready at http://localhost:%BACKEND_PORT%

:: Start frontend in new window with explicit port and target mapping
echo [AIQ] Starting frontend on port %FRONTEND_PORT%...
cd /d "%FRONTEND_DIR%"
start "AIQ Frontend" cmd /k "cd /d "%FRONTEND_DIR%" && set "BACKEND_PORT=%BACKEND_PORT%" && set "AION_PORT=%BACKEND_PORT%" && set "FRONTEND_PORT=%FRONTEND_PORT%" && pnpm run dev --host --port %FRONTEND_PORT%"

echo.
echo ================================================
echo   AIQ is running
echo   Backend  : http://localhost:%BACKEND_PORT%
echo   Frontend : http://localhost:%FRONTEND_PORT%
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
call :allocate_ports
echo [AIQ] Starting backend on port %BACKEND_PORT%...
set "AION_PORT=%BACKEND_PORT%"
python aion_api.py
goto :eof

:: ── Frontend only ─────────────────────────────────────────────────────────────
:cmd_frontend
if not exist "%FRONTEND_DIR%" (
    echo [AIQ ERROR] Frontend not found: %FRONTEND_DIR%
    exit /b 1
)
call :read_ports
if defined PORT_OVERRIDE set "FRONTEND_PORT=%PORT_OVERRIDE%"
cd /d "%FRONTEND_DIR%"
echo [AIQ] Starting frontend on port %FRONTEND_PORT% (mapping backend %BACKEND_PORT%)...
set "BACKEND_PORT=%BACKEND_PORT%"
set "AION_PORT=%BACKEND_PORT%"
set "FRONTEND_PORT=%FRONTEND_PORT%"
pnpm run dev --host --port %FRONTEND_PORT%
goto :eof

:: ── Stop ──────────────────────────────────────────────────────────────────────
:cmd_stop
echo [AIQ] Stopping AIQ services...
call :read_ports

:: Safely stop backend processes: only terminate if process image is Python
for %%P in (%BACKEND_PORT% %DEFAULT_BACKEND_PORT%) do (
    for /f "tokens=5" %%I in ('netstat -ano ^| findstr ":%%P "') do (
        if not "%%I"=="0" (
            for /f "tokens=1 delims=," %%N in ('tasklist /fi "PID eq %%I" /fo csv /nh 2^>nul') do (
                set "PNAME=%%~N"
                if /i "!PNAME!"=="python.exe" (
                    taskkill /F /PID %%I >nul 2>&1
                    echo [AIQ] Backend stopped (PID %%I, port %%P)
                ) else if /i "!PNAME!"=="py.exe" (
                    taskkill /F /PID %%I >nul 2>&1
                    echo [AIQ] Backend stopped (PID %%I, port %%P)
                ) else (
                    echo [AIQ] Notice: Port %%P PID %%I is '!PNAME!' (not Python) - skipping kill.
                )
            )
        )
    )
)

:: Safely stop frontend dev server: only terminate if process image is Node
for %%P in (%FRONTEND_PORT% 5174 5173) do (
    for /f "tokens=5" %%I in ('netstat -ano ^| findstr ":%%P "') do (
        if not "%%I"=="0" (
            for /f "tokens=1 delims=," %%N in ('tasklist /fi "PID eq %%I" /fo csv /nh 2^>nul') do (
                set "PNAME=%%~N"
                if /i "!PNAME!"=="node.exe" (
                    taskkill /F /PID %%I >nul 2>&1
                    echo [AIQ] Frontend stopped (PID %%I, port %%P)
                ) else (
                    echo [AIQ] Notice: Port %%P PID %%I is '!PNAME!' (not Node.js) - skipping kill.
                )
            )
        )
    )
)

:: Clean up ports.env on clean stop
if exist "%PID_DIR%\ports.env" del "%PID_DIR%\ports.env" >nul 2>&1

echo [AIQ] Done
goto :eof

:: ── Status ────────────────────────────────────────────────────────────────────
:cmd_status
echo.
echo [AIQ] Status Check
echo.
call :read_ports

:: Backend (verifies /api/health)
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:%BACKEND_PORT%/api/health' -TimeoutSec 2 -UseBasicParsing; if ($r.StatusCode -eq 200) { Write-Host '  Backend   [OK]  running  (port %BACKEND_PORT%)' -ForegroundColor Green } else { Write-Host '  Backend   [WARN] responded with status ' $r.StatusCode -ForegroundColor Yellow } } catch { try { (New-Object Net.Sockets.TcpClient('127.0.0.1',%BACKEND_PORT%)).Close(); Write-Host '  Backend   [OK]  port %BACKEND_PORT% open' -ForegroundColor Green } catch { Write-Host '  Backend   [DOWN] stopped (port %BACKEND_PORT%)' -ForegroundColor Red } }"

:: Ollama
powershell -NoProfile -Command "try { (New-Object Net.Sockets.TcpClient('127.0.0.1',11434)).Close(); Write-Host '  Ollama    [OK]  running  (port 11434)' -ForegroundColor Green } catch { Write-Host '  Ollama    [DOWN] stopped (port 11434)' -ForegroundColor Red }"

:: Frontend
powershell -NoProfile -Command "try { (New-Object Net.Sockets.TcpClient('127.0.0.1',%FRONTEND_PORT%)).Close(); Write-Host '  Frontend  [OK]  running  (port %FRONTEND_PORT%)' -ForegroundColor Green } catch { Write-Host '  Frontend  [DOWN] stopped (port %FRONTEND_PORT%)' -ForegroundColor Red }"

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
    powershell -NoProfile -Command "Get-Content '%BACKEND_LOG%' -Wait"
) else (
    echo [AIQ ERROR] No log file at %BACKEND_LOG%
    echo Start AIQ first: aiq
)
goto :eof

:: ── Help ──────────────────────────────────────────────────────────────────────
:cmd_help
call :banner
echo Usage: aiq [command] [port]
echo.
echo Commands:
echo   (none)     Start backend + frontend (auto-allocating free ports)
echo   backend    Start backend only (e.g. aiq backend 8105)
echo   frontend   Start frontend only (e.g. aiq frontend 5175)
echo   stop       Safely stop AIQ processes (verifies process identity)
echo   status     Show running status and verified ports
echo   logs       Tail backend logs
echo   model      Show model configuration
echo   help       Show this help
echo.
echo Environment overrides:
echo   AION_PORT=8100      Preferred backend port
echo   FRONTEND_PORT=5174  Preferred frontend port
echo   AION_DEVICE=laptop^|desktop^|server
echo.
echo Examples:
echo   aiq
echo   aiq 8105
echo   aiq backend 8102
echo   aiq status
echo   aiq stop
goto :eof

endlocal
