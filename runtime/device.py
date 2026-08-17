# runtime/device.py

import os
import sys
import platform
import subprocess
import urllib.request
from typing import Dict, Any

def get_cpu_info() -> str:
    """Get CPU model name in a cross-platform manner."""
    try:
        if platform.system() == "Windows":
            # Command to get CPU on Windows
            out = subprocess.check_output(["wmic", "cpu", "get", "name"], text=True)
            lines = [line.strip() for line in out.split("\n") if line.strip()]
            if len(lines) > 1:
                return lines[1]
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"

def get_ram_info() -> float:
    """Get total system memory in GB."""
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"], text=True)
            lines = [line.strip() for line in out.split("\n") if line.strip()]
            if len(lines) > 1:
                return float(lines[1]) / (1024**3)
        elif platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        return float(line.split()[1]) / (1024**2)
    except Exception:
        pass
    return 16.0  # Safe default fallback

def get_gpu_info() -> str:
    """Detect available GPU devices (Intel Arc, Nvidia, etc.)."""
    gpus = []
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(["wmic", "path", "win32_VideoController", "get", "name"], text=True)
            lines = [line.strip() for line in out.split("\n") if line.strip()][1:]
            for line in lines:
                if line:
                    gpus.append(line)
        elif platform.system() == "Linux":
            out = subprocess.check_output(["lspci"], text=True)
            for line in out.split("\n"):
                if "VGA" in line or "3D" in line or "Display" in line:
                    gpus.append(line.split(":")[-1].strip())
    except Exception:
        pass
    
    if gpus:
        # Prioritize Intel Arc or Integrated graphics details
        for gpu in gpus:
            if "arc" in gpu.lower() or "intel" in gpu.lower():
                return gpu
        return gpus[0]
    return "Intel Integrated Graphics"

def check_openvino() -> bool:
    """Check if OpenVINO is available."""
    try:
        import openvino
        return True
    except ImportError:
        return False

def check_llamacpp() -> bool:
    """Check if llama.cpp python binding or executable is available."""
    try:
        import llama_cpp
        return True
    except ImportError:
        pass
    
    # Check for CLI executable in PATH
    for ext in ["", ".exe"]:
        try:
            subprocess.run(["llama-cli" + ext, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            pass
    return False

def check_ollama() -> bool:
    """Check if local Ollama port is accessible."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=1.0) as response:
            return response.status == 200
    except Exception:
        return False

def get_device_report() -> Dict[str, Any]:
    """Compile and return the complete device capability report."""
    return {
        "cpu": get_cpu_info(),
        "ram_gb": round(get_ram_info(), 2),
        "gpu": get_gpu_info(),
        "openvino_available": check_openvino(),
        "llamacpp_available": check_llamacpp(),
        "ollama_available": check_ollama(),
    }

if __name__ == "__main__":
    report = get_device_report()
    print("=== AION Hardware Detection ===")
    for k, v in report.items():
        print(f"{k:20}: {v}")
