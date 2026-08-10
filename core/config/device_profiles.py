"""
AION Device Hardware Profiles
=============================
Defines hardware profiles mapping device type to model authority and RAM requirements.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class DeviceProfileSpec:
    name: str
    model: str
    min_ram_gb: float
    description: str


PROFILES: Dict[str, DeviceProfileSpec] = {
    "server": DeviceProfileSpec("server", "qwen2.5:14b", 32.0, "Production GPU Server (A100 / L40)"),
    "desktop": DeviceProfileSpec("desktop", "qwen2.5:7b", 12.0, "High-end Workstation / Desktop"),
    "laptop": DeviceProfileSpec("laptop", "qwen2.5:3b", 6.0, "Standard Laptop / Notebook"),
    "light": DeviceProfileSpec("light", "qwen2.5:1.5b", 0.0, "CI / Testing / Low-RAM Environment"),
}
