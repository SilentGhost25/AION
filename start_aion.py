#!/usr/bin/env python3
"""
AION Complete Startup Manager
Handles Ollama + Flask API in correct order.
Run once: python start_aion.py
"""

import subprocess
import sys
import time
import socket
import os
import signal
import platform
import requests
from pathlib import Path

# Force UTF-8 encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ═══════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════

CONFIG = {
    "ollama_port":     11434,
    "api_port":        8100,
    "model":           "qwen2.5:7b",
    "api_script":      "aion_api.py",
    "startup_timeout": 30,     # seconds to wait
}

ROOT = Path(__file__).parent.resolve()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════

def is_port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def print_status(label: str, ok: bool, detail: str = ""):
    icon = "[OK]" if ok else "[FAIL]"
    line = f"  {icon} {label}"
    if detail:
        line += f": {detail}"
    print(line)


def wait_for_port(port: int, timeout: int, label: str) -> bool:
    """
    Wait for a port to become available.
    Shows progress.
    """
    print(f"  Waiting for {label}...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        if is_port_open("localhost", port):
            print(" [READY]")
            return True
        print(".", end="", flush=True)
        time.sleep(1)
    print(f" [TIMEOUT after {timeout}s]")
    return False


# ═══════════════════════════════════════════════════
# DEPENDENCY CHECKS
# ═══════════════════════════════════════════════════

def check_dependencies() -> bool:
    """
    Verify all required packages are installed.
    """
    print("\n" + "="*55)
    print("  Checking Dependencies")
    print("="*55)

    all_ok = True

    REQUIRED = {
        "flask":        "flask",
        "flask_cors":   "flask-cors",
        "requests":     "requests",
        "fitz":         "pymupdf",
    }

    for import_name, install_name in REQUIRED.items():
        try:
            __import__(import_name)
            print_status(f"Package: {install_name}", True)
        except ImportError:
            print_status(
                f"Package: {install_name}", False,
                f"Run: pip install {install_name}"
            )
            all_ok = False

    # Check urllib3 version
    try:
        import urllib3
        v = tuple(int(x) for x in urllib3.__version__.split(".")[:2] if x.isdigit())
        if v >= (2, 0):
            print_status(
                "urllib3 version", False,
                f"v{urllib3.__version__} conflicts with requests. Run: pip install 'urllib3<2'"
            )
        else:
            print_status("urllib3 version", True, urllib3.__version__)
    except Exception:
        pass

    return all_ok


# ═══════════════════════════════════════════════════
# OLLAMA MANAGEMENT
# ═══════════════════════════════════════════════════

def find_ollama_executable() -> str:
    """Find the Ollama executable on this system."""
    system = platform.system()

    if system == "Windows":
        candidates = [
            rf"C:\Users\{os.getenv('USERNAME', '')}\AppData\Local\Programs\Ollama\ollama.exe",
            r"C:\Program Files\Ollama\ollama.exe",
            "ollama",
        ]
        for path in candidates:
            try:
                result = subprocess.run(
                    [path, "--version"],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return "ollama"
    else:
        return "ollama"


def is_ollama_running() -> bool:
    """Check if Ollama API is responding."""
    try:
        r = requests.get(
            f"http://localhost:{CONFIG['ollama_port']}/api/tags",
            timeout=3
        )
        return r.status_code == 200
    except Exception:
        return False


def start_ollama() -> subprocess.Popen | None:
    """
    Start Ollama as a background process.
    Returns the process handle or None if failed.
    """
    if is_ollama_running():
        print_status("Ollama", True, "already running")
        return None

    print("  Starting Ollama...")
    ollama_exe = find_ollama_executable()

    try:
        if platform.system() == "Windows":
            proc = subprocess.Popen(
                [ollama_exe, "serve"],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        else:
            proc = subprocess.Popen(
                [ollama_exe, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        if wait_for_port(CONFIG["ollama_port"], CONFIG["startup_timeout"], "Ollama"):
            return proc
        else:
            print("  Ollama did not start in time.\n  Try manually: ollama serve")
            return None

    except FileNotFoundError:
        print(f"  Ollama executable not found.\n  Install from: https://ollama.com")
        return None


def ensure_model_available(model: str) -> bool:
    """
    Check if the required model is installed.
    """
    try:
        r = requests.get(
            f"http://localhost:{CONFIG['ollama_port']}/api/tags",
            timeout=5
        )
        models = r.json().get("models", [])
        model_names = [m["name"] for m in models]

        model_base = model.split(":")[0]
        exists = any(model_base in name for name in model_names)

        if exists:
            print_status(f"Model: {model}", True)
            return True

        print_status(f"Model: {model}", False, "not installed")
        return False

    except Exception as e:
        print_status("Model check", False, str(e))
        return False


# ═══════════════════════════════════════════════════
# FLASK API MANAGEMENT
# ═══════════════════════════════════════════════════

def start_flask_api() -> subprocess.Popen | None:
    """Start the AION Flask API."""
    if is_port_open("localhost", CONFIG["api_port"]):
        print_status(f"API port {CONFIG['api_port']}", True, "already running")
        return None

    api_script = Path(CONFIG["api_script"])
    if not api_script.exists():
        print_status("API script", False, f"{api_script} not found")
        return None

    print("  Starting Flask API...")

    try:
        if platform.system() == "Windows":
            proc = subprocess.Popen(
                [sys.executable, str(api_script)],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        else:
            proc = subprocess.Popen(
                [sys.executable, str(api_script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        if wait_for_port(CONFIG["api_port"], CONFIG["startup_timeout"], "Flask API"):
            return proc
        else:
            print("  Flask API did not start.")
            return None

    except Exception as e:
        print(f"  Failed to start API: {e}")
        return None


# ═══════════════════════════════════════════════════
# HEALTH VERIFICATION
# ═══════════════════════════════════════════════════

def verify_full_stack() -> dict:
    """
    Verify all components are working together.
    Returns health report.
    """
    print("\n" + "="*55)
    print("  System Health Check")
    print("="*55)

    health = {}

    # 1. Ollama API
    ollama_ok = is_ollama_running()
    print_status("Ollama API", ollama_ok, f"port {CONFIG['ollama_port']}")
    health["ollama"] = ollama_ok

    # 2. Flask API
    flask_ok = is_port_open("localhost", CONFIG["api_port"])
    print_status("Flask API", flask_ok, f"port {CONFIG['api_port']}")
    health["flask"] = flask_ok

    # 3. AION health endpoint
    try:
        r = requests.get(
            f"http://localhost:{CONFIG['api_port']}/api/health",
            timeout=5
        )
        health_data = r.json()
        aion_ok = (health_data.get("status") in ["ok", "degraded"])
        print_status("AION /api/health", aion_ok, str(health_data.get("status", "unknown")))
        health["aion"] = aion_ok
    except Exception as e:
        print_status("AION /api/health", False, str(e))
        health["aion"] = False

    return health


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════

def main():
    print("\n" + "="*55)
    print("  AION Startup Manager")
    print("="*55)

    processes = []

    try:
        # Stage 1: Dependencies
        if not check_dependencies():
            print("\nFix dependency errors first.")
            sys.exit(1)

        # Stage 2: Start Ollama
        print("\n" + "="*55)
        print("  Starting Ollama")
        print("="*55)
        ollama_proc = start_ollama()
        if ollama_proc:
            processes.append(ollama_proc)

        if not is_ollama_running():
            print("\nCannot proceed without Ollama.\n  Manual fix:\n  1. Open a terminal\n  2. Run: ollama serve")
            sys.exit(1)

        # Stage 3: Ensure model available
        print("\n" + "="*55)
        print("  Checking Model")
        print("="*55)
        ensure_model_available(CONFIG["model"])

        # Stage 4: Start Flask API
        print("\n" + "="*55)
        print("  Starting Flask API")
        print("="*55)
        flask_proc = start_flask_api()
        if flask_proc:
            processes.append(flask_proc)

        # Stage 5: Verify everything
        time.sleep(2)
        health = verify_full_stack()

        all_ok = all(health.values())

        print("\n" + "="*55)
        if all_ok:
            print("  AION is ready!")
            print("="*55)
            print(f"  API:      http://localhost:{CONFIG['api_port']}")
            print(f"  Health:   http://localhost:{CONFIG['api_port']}/api/health")
            print(f"  Frontend: http://localhost:5173")
            print("\n  Press Ctrl+C to stop AION\n")
        else:
            print("  AION started with warnings")
            print("="*55)

    except KeyboardInterrupt:
        print("\n\n[AION] Shutting down...")
        for proc in processes:
            try:
                proc.terminate()
            except Exception:
                pass
        print("[AION] Stopped.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
