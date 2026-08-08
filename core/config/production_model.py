"""
AION Model Configuration Authority
====================================
Single source of truth for model selection.
Every component imports from here. Nothing else defines a model.

Resolution order (highest to lowest priority):
  1. AION_MODEL environment variable        (manual override)
  2. config.json file in AION root          (user config)
  3. AION_DEVICE environment variable       (device profile)
  4. Auto-detect from available RAM         (device detection)
  5. Production default                     (qwen2.5:14b)

Device profiles:
  server    → qwen2.5:14b   (L40 / A100 / production GPU server)
  desktop   → qwen2.5:7b    (RTX 3090 / high-end workstation)
  laptop    → qwen2.5:3b    (Intel Arc / 16GB RAM laptop)
  light     → qwen2.5:1.5b  (CI / testing / very low RAM)
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List


# ── Model Profiles ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelProfile:
    production:  str = "qwen2.5:14b"
    desktop:     str = "qwen2.5:7b"
    laptop:      str = "qwen2.5:3b"
    lightweight: str = "qwen2.5:1.5b"


PROFILE = ModelProfile()

# Maps AION_DEVICE value → model name
DEVICE_MAP: Dict[str, str] = {
    "server":      PROFILE.production,
    "production":  PROFILE.production,
    "desktop":     PROFILE.desktop,
    "workstation": PROFILE.desktop,
    "laptop":      PROFILE.laptop,
    "notebook":    PROFILE.laptop,
    "light":       PROFILE.lightweight,
    "ci":          PROFILE.lightweight,
    "test":        PROFILE.lightweight,
}

# RAM thresholds for auto-detection (GB)
RAM_THRESHOLDS: List[Tuple[float, str]] = [
    (32.0, PROFILE.production),   # ≥ 32 GB  → 14B
    (12.0, PROFILE.desktop),      # ≥ 12 GB  → 7B
    (6.0,  PROFILE.laptop),       # ≥  6 GB  → 3B
    (0.0,  PROFILE.lightweight),  # < 6 GB   → 1.5B
]

# Where to look for user config
_CONFIG_PATHS = [
    Path(__file__).parent.parent.parent / "config.json",   # AION/config.json
    Path(__file__).parent.parent.parent / "aion_config.yaml",
    Path.home() / ".aion" / "config.json",
]


# ── Resolution Source (for health endpoint reporting) ────────────────────────

_last_resolution: dict = {}


def get_production_model() -> str:
    """
    Resolve and return the model name for the current environment.
    Thread-safe, side-effect-free, never raises.

    Always import this function. Never hard-code a model name.
    """
    global _last_resolution

    # Priority 1: AION_MODEL environment variable (manual override)
    env_model = os.environ.get("AION_MODEL", "").strip()
    if env_model:
        _last_resolution = {
            "resolved_model": env_model,
            "source": "env_AION_MODEL",
            "device": os.environ.get("AION_DEVICE", "unset"),
        }
        return env_model

    # Priority 2: config.json
    config_model = _read_config_file()
    if config_model:
        _last_resolution = {
            "resolved_model": config_model,
            "source": "config_file",
            "device": os.environ.get("AION_DEVICE", "unset"),
        }
        return config_model

    # Priority 3: AION_DEVICE environment variable
    device = os.environ.get("AION_DEVICE", "").strip().lower()
    if device and device in DEVICE_MAP:
        model = DEVICE_MAP[device]
        _last_resolution = {
            "resolved_model": model,
            "source": "env_AION_DEVICE",
            "device": device,
        }
        return model

    # Priority 4: Auto-detect from available RAM
    detected_device, auto_model = _auto_detect()
    if auto_model:
        _last_resolution = {
            "resolved_model": auto_model,
            "source": "auto_detect_ram",
            "device": detected_device,
        }
        return auto_model

    # Priority 5: Production default
    _last_resolution = {
        "resolved_model": PROFILE.production,
        "source": "production_default",
        "device": "server",
    }
    return PROFILE.production


def get_resolution_info() -> dict:
    """
    Return information about how the current model was resolved.
    Used by the health endpoint to report model source.

    Example:
        {
            "resolved_model": "qwen2.5:3b",
            "source": "env_AION_DEVICE",
            "device": "laptop"
        }
    """
    if not _last_resolution:
        get_production_model()   # trigger resolution
    return dict(_last_resolution)


def get_model_for_device(device: str) -> str:
    """Return the model for a specific device name."""
    return DEVICE_MAP.get(device.lower(), PROFILE.production)


def is_production() -> bool:
    """Return True if running in production (server) mode."""
    info = get_resolution_info()
    return info.get("device") in ("server", "production")


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _read_config_file() -> Optional[str]:
    """Read model from user config.json if it exists."""
    for path in _CONFIG_PATHS:
        try:
            if path.exists() and path.suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                model = data.get("model") or data.get("aion_model")
                if model and isinstance(model, str):
                    return model.strip()
        except Exception:
            pass
    return None


def _auto_detect() -> Tuple[str, Optional[str]]:
    """
    Auto-detect device type from available system RAM.
    Returns (device_label, model_name).
    """
    try:
        import psutil
        free_gb = psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        # Fallback without psutil
        try:
            if platform.system() == "Windows":
                import ctypes
                stat = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(
                    ctypes.byref(stat)
                )
                free_gb = stat.value / (1024 ** 2)
            else:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if "MemAvailable" in line:
                            free_gb = int(line.split()[1]) / (1024 ** 2)
                            break
                    else:
                        return "unknown", None
        except Exception:
            return "unknown", None

    for threshold, model in RAM_THRESHOLDS:
        if free_gb >= threshold:
            if free_gb >= 32:
                device = "server"
            elif free_gb >= 12:
                device = "desktop"
            elif free_gb >= 6:
                device = "laptop"
            else:
                device = "light"
            return device, model

    return "unknown", PROFILE.lightweight
